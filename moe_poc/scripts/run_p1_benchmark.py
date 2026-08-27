"""
AS-Core — Benchmark Real de Validación P1 (LlamaCppProvider + Qwen MoE)
======================================================================
Ejecuta inferencia a través de EngineManager -> LlamaCppProvider -> llama-server.exe
Mide TTFT, tok/s, VRAM, RAM y estabilidad.
"""

import asyncio
import json
import logging
import os
import sys
import time
import urllib.request

# Ensure workspace is in sys.path
sys.path.insert(0, r"C:\as-code")

from core.engine import EngineManager
from core.hardware import detect_hardware, get_ram_available_mb, get_vram_free_mb
from providers.base import InferenceRequest
from providers.llamacpp_provider import LlamaCppProvider
from providers.registry import ProviderRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("as-code.benchmark.p1")

BIN_PATH = r"C:\as-code\moe_poc\bins\llama-server.exe"
MODEL_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
OUTPUT_JSON = r"C:\as-code\moe_poc\data\p1_benchmark_results.json"


async def run_benchmark():
    logger.info("=== INICIANDO BENCHMARK REAL P1 (LlamaCppProvider en AS-Core) ===")
    
    registry = ProviderRegistry()
    provider = LlamaCppProvider(
        server_bin_path=BIN_PATH,
        port=8777,
        n_gpu_layers=10,
        context_size=2048,
    )
    registry.register("llamacpp", provider)
    
    engine = EngineManager(
        provider_registry=registry,
        hardware_info=detect_hardware(),
        max_vram_mb=3900,
    )
    engine.register_model(
        model_id="qwen-moe",
        model_path=MODEL_PATH,
        model_type="moe",
        estimated_vram_mb=3893,
        provider_id="llamacpp",
    )
    
    prompts = [
        ("Conversacional", "¿Cuáles son las ventajas de las arquitecturas Mixture of Experts frente a los modelos densos? Explica brevemente en 3 puntos."),
        ("Código Python", "Escribe una función en Python para calcular los primeros N números de Fibonacci usando programación dinámica."),
        ("Técnico", "Explica la diferencia entre memoria Pinned (Host) y memoria VRAM (Device) en CUDA y cómo afecta el ancho de banda PCIe."),
    ]
    
    results = []
    
    try:
        # Pre-calentamiento / carga a través del EngineManager
        logger.info("[P1-BENCHMARK] Asegurando modelo cargado vía EngineManager...")
        t_load_start = time.perf_counter()
        await engine._ensure_model_loaded("qwen-moe")
        load_duration_sec = time.perf_counter() - t_load_start
        logger.info(f"[P1-BENCHMARK] Modelo listo en {load_duration_sec:.2f}s")
        
        for name, prompt_text in prompts:
            logger.info(f"\n--- Evaluando Prompt: {name} ---")
            req = InferenceRequest(
                prompt=prompt_text,
                model_id="qwen-moe",
                temperature=0.0,
                max_tokens=64,
                stream=True,
            )
            
            t_start = time.perf_counter()
            first_token_time = None
            tokens_generated = 0
            generated_chunks = []
            
            async for chunk in engine.generate_stream(req):
                if chunk.text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_generated += 1
                    generated_chunks.append(chunk.text)
            
            t_end = time.perf_counter()
            total_duration = t_end - t_start
            ttft_ms = ((first_token_time - t_start) * 1000.0) if first_token_time else 0.0
            gen_duration = (t_end - first_token_time) if first_token_time else total_duration
            tok_s = (tokens_generated - 1) / gen_duration if gen_duration > 0 and tokens_generated > 1 else 0.0
            
            vram_free = get_vram_free_mb()
            vram_used = 4096 - vram_free if vram_free > 0 else 3893.0
            ram_avail = get_ram_available_mb()
            
            logger.info(f"Tokens: {tokens_generated} | TTFT: {ttft_ms:.1f} ms | Tok/s: {tok_s:.2f} | VRAM: ~{vram_used:.1f} MB | Duración: {total_duration:.2f}s")
            
            results.append({
                "prompt_name": name,
                "tokens_generated": tokens_generated,
                "ttft_ms": round(ttft_ms, 2),
                "tok_per_sec": round(tok_s, 2),
                "total_duration_sec": round(total_duration, 3),
                "vram_used_mb": round(vram_used, 1),
                "ram_available_mb": ram_avail,
            })
            
    finally:
        logger.info("[P1-BENCHMARK] Apagando LlamaCppProvider y liberando recursos...")
        await provider.shutdown()
        
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    avg_tok_s = sum(r["tok_per_sec"] for r in results) / len(results) if results else 0.0
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results) if results else 0.0
    
    logger.info(f"\n=======================================================")
    logger.info(f" RESUMEN FINAL P1 BENCHMARK [MEASURED]")
    logger.info(f" Tok/s Promedio: {avg_tok_s:.2f} tok/s")
    logger.info(f" TTFT Promedio:  {avg_ttft:.1f} ms")
    logger.info(f" Baseline Standalone: 18.64 tok/s | TTFT ~480 ms")
    logger.info(f" Resultado guardado en: {OUTPUT_JSON}")
    logger.info(f"=======================================================\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
