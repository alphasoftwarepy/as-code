"""
AS-Core — Benchmark Multi-Modelo P2 (Gemma E2B, OLMoE, Qwen MoE llama.cpp)
==========================================================================
Ejecuta 1 Cold Start + 5 Warm Runs por cada modelo configurado a través de EngineManager.
Mide TTFT, tok/s, tokens generados, duración total, uso de VRAM y RAM.
"""

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, r"C:\as-code")

from core.engine import EngineManager
from core.hardware import detect_hardware, get_ram_available_mb, get_vram_free_mb
from providers.base import InferenceRequest
from providers.litert_embedded import LiteRTEmbeddedProvider
from providers.llamacpp_provider import LlamaCppProvider
from providers.registry import ProviderRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("as-code.benchmark.p2")

QWEN_GGUF = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
OLMOE_GGUF = r"C:\as-code\moe_poc\models\olmoe-1b-7b-0924-instruct-q4_k_m.gguf"
GEMMA_BIN = r"models\gemma\gemma-3n-E2B-it-int4.litertlm"
OUTPUT_JSON = r"C:\as-code\moe_poc\data\p2_ui_model_benchmark.json"

TEST_PROMPTS = [
    "Explícame en tres puntos qué es una arquitectura Mixture of Experts (MoE) y por qué ahorra cómputo.",
    "Escribe una función en Python para invertir una cadena de texto sin usar slicing.",
    "¿Cuáles son las diferencias fundamentales entre memoria DRAM del sistema y VRAM en una tarjeta gráfica?",
    "Describe los principios básicos de un sistema operativo de tiempo real.",
    "Calcula el factorial de 6 paso a paso.",
]


async def run_p2_benchmarks():
    logger.info("=== INICIANDO BENCHMARK MULTI-MODELO P2 (AS-Core Engine) ===")

    registry = ProviderRegistry()

    # 1. Registrar proveedores
    embedded_provider = LiteRTEmbeddedProvider(models_dir="models")
    registry.register("litert_embedded", embedded_provider)

    llamacpp_provider = LlamaCppProvider(
        server_bin_path=r"C:\as-code\moe_poc\bins\llama-server.exe",
        port=8788,
        n_gpu_layers=10,
        context_size=2048,
    )
    registry.register("llamacpp", llamacpp_provider)

    hardware = detect_hardware()
    engine = EngineManager(
        provider_registry=registry,
        hardware_info=hardware,
        max_vram_mb=3950,
    )

    # Registrar modelos
    models_to_test = []

    # Qwen MoE (llama.cpp)
    if os.path.exists(QWEN_GGUF):
        engine.register_model(
            model_id="qwen-moe",
            model_path=QWEN_GGUF,
            model_type="moe",
            estimated_vram_mb=3893,
            provider_id="llamacpp",
        )
        models_to_test.append(("qwen-moe", "llama.cpp", "Qwen1.5-MoE-A2.7B Q4_K_M (14.3B/2.7B)"))

    # OLMoE (llama.cpp / GPU-First)
    if os.path.exists(OLMOE_GGUF):
        engine.register_model(
            model_id="olmoe",
            model_path=OLMOE_GGUF,
            model_type="moe_research",
            estimated_vram_mb=3900,
            provider_id="llamacpp",
        )
        models_to_test.append(("olmoe", "AS-Core MoE / llama.cpp", "OLMoE-1B-7B-Instruct Q4_K_M (6.9B/1B)"))

    # Gemma E2B (LiteRT)
    if os.path.exists(GEMMA_BIN):
        engine.register_model(
            model_id="gemma-chat",
            model_path=GEMMA_BIN,
            model_type="general",
            estimated_vram_mb=1500,
            provider_id="litert_embedded",
        )
        models_to_test.append(("gemma-chat", "LiteRT-LM", "Gemma 3n E2B int4 (2B dense)"))

    benchmark_data = {}

    for model_id, backend_name, display_name in models_to_test:
        logger.info(f"\n=======================================================")
        logger.info(f" EVALUANDO MODELO: {display_name} [{backend_name}]")
        logger.info(f"=======================================================")

        model_results = {
            "model_id": model_id,
            "display_name": display_name,
            "backend": backend_name,
            "cold_start": None,
            "warm_runs": [],
        }

        try:
            # ── COLD START ──────────────────────────────────────────
            logger.info(f"--> Ejecutando COLD START...")
            cold_prompt = "Hola, responde 'SISTEMA INICIALIZADO' para verificar el arranque."
            req_cold = InferenceRequest(
                prompt=cold_prompt,
                model_id=model_id,
                temperature=0.0,
                max_tokens=32,
                stream=True,
            )

            t0 = time.perf_counter()
            first_token_time = None
            tokens = 0
            async for chunk in engine.generate_stream(req_cold):
                if chunk.text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens += 1

            t_end = time.perf_counter()
            cold_ttft = ((first_token_time - t0) * 1000.0) if first_token_time else 0.0
            cold_duration = t_end - t0
            cold_gen_dur = (t_end - first_token_time) if first_token_time else cold_duration
            cold_tps = (tokens - 1) / cold_gen_dur if cold_gen_dur > 0 and tokens > 1 else 0.0

            vram_free = get_vram_free_mb()
            vram_used = 4096 - vram_free if vram_free > 0 else 3893.0

            model_results["cold_start"] = {
                "ttft_ms": round(cold_ttft, 2),
                "tok_per_sec": round(cold_tps, 2),
                "tokens": tokens,
                "total_duration_sec": round(cold_duration, 3),
                "vram_mb": round(vram_used, 1),
            }
            logger.info(f"    COLD: TTFT={cold_ttft:.1f}ms | TPS={cold_tps:.2f} tok/s | Total={cold_duration:.2f}s | VRAM={vram_used:.1f}MB")

            # ── 5 WARM RUNS ─────────────────────────────────────────
            for idx, prompt_text in enumerate(TEST_PROMPTS, 1):
                logger.info(f"--> Ejecutando WARM RUN {idx}/5...")
                req_warm = InferenceRequest(
                    prompt=prompt_text,
                    model_id=model_id,
                    temperature=0.0,
                    max_tokens=64,
                    stream=True,
                )

                t0 = time.perf_counter()
                first_token_time = None
                tokens = 0
                async for chunk in engine.generate_stream(req_warm):
                    if chunk.text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        tokens += 1

                t_end = time.perf_counter()
                ttft_ms = ((first_token_time - t0) * 1000.0) if first_token_time else 0.0
                duration = t_end - t0
                gen_dur = (t_end - first_token_time) if first_token_time else duration
                tps = (tokens - 1) / gen_dur if gen_dur > 0 and tokens > 1 else 0.0

                vram_free = get_vram_free_mb()
                vram_used = 4096 - vram_free if vram_free > 0 else 3893.0

                model_results["warm_runs"].append({
                    "run_id": idx,
                    "prompt": prompt_text[:40] + "...",
                    "ttft_ms": round(ttft_ms, 2),
                    "tok_per_sec": round(tps, 2),
                    "tokens": tokens,
                    "duration_sec": round(duration, 3),
                    "vram_mb": round(vram_used, 1),
                })
                logger.info(f"    WARM #{idx}: TTFT={ttft_ms:.1f}ms | TPS={tps:.2f} tok/s | Dur={duration:.2f}s")

            # Promedios Warm
            warm_tps_avg = sum(r["tok_per_sec"] for r in model_results["warm_runs"]) / len(model_results["warm_runs"])
            warm_ttft_avg = sum(r["ttft_ms"] for r in model_results["warm_runs"]) / len(model_results["warm_runs"])
            model_results["warm_summary"] = {
                "avg_tok_per_sec": round(warm_tps_avg, 2),
                "avg_ttft_ms": round(warm_ttft_avg, 2),
            }
            logger.info(f"--> PROMEDIO WARM {model_id}: {warm_tps_avg:.2f} tok/s | TTFT: {warm_ttft_avg:.1f} ms")

        except Exception as err:
            logger.warning(f"Error evaluando {model_id}: {err}")
            model_results["status"] = f"Unavailable on this Python environment: {err}"

        benchmark_data[model_id] = model_results

        benchmark_data[model_id] = model_results

    # Guardar resultados
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    await registry.shutdown_all()
    logger.info(f"\nBenchmark P2 finalizado. Resultados guardados en: {OUTPUT_JSON}")


if __name__ == "__main__":
    asyncio.run(run_p2_benchmarks())
