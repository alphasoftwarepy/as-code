"""
AS Core — Inference & Residency Benchmark
Runs programmatically to validate cold starts, warm starts, swapping latency,
and idle timeout unloads.
"""

import asyncio
import logging
import time
import sys
import os

# Set up PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import get_settings
from core.engine import EngineManager
from core.hardware import detect_hardware
from providers.litert_cli import LiteRTCLIProvider
from providers.registry import ProviderRegistry
from providers.base import InferenceRequest

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark")

async def run_inference(engine: EngineManager, model_id: str, prompt: str, label: str):
    logger.info(f"--- Running Inference: {label} ({model_id}) ---")
    request = InferenceRequest(
        prompt=prompt,
        model_id=model_id,
        temperature=0.7,
        max_tokens=20,
    )
    
    start_time = time.perf_counter()
    first_token_time = None
    total_tokens = 0
    generated_text = ""

    async for chunk in engine.generate_stream(request):
        if first_token_time is None and chunk.text:
            first_token_time = time.perf_counter()
        if chunk.text:
            generated_text += chunk.text
            total_tokens += 1
            print(chunk.text, end="", flush=True)

    print() # Newline
    end_time = time.perf_counter()
    
    total_elapsed = end_time - start_time
    time_to_first = (first_token_time - start_time) if first_token_time else 0
    generation_time = total_elapsed - time_to_first
    tps = total_tokens / generation_time if generation_time > 0 else 0

    logger.info(f"Results for {label}:")
    logger.info(f"  - Total Elapsed: {total_elapsed:.3f}s")
    logger.info(f"  - Time to First Token: {time_to_first:.3f}s")
    logger.info(f"  - Generation Time: {generation_time:.3f}s")
    logger.info(f"  - Est. Tokens: {total_tokens} (TPS: {tps:.2f})")
    return {
        "total_elapsed": total_elapsed,
        "time_to_first": time_to_first,
        "tps": tps,
    }

async def main():
    settings = get_settings()
    hardware = detect_hardware()
    
    # Configure custom short unload timeout to test unload loop quickly
    settings.model_unload_timeout_sec = 10.0
    settings.model_absolute_lifetime_sec = 20.0

    registry = ProviderRegistry()
    registry.register("litert_cli", LiteRTCLIProvider(
        cli_path=settings.litert_cli_path,
        default_backend=settings.litert_backend,
        enable_speculative_decoding=settings.enable_speculative_decoding,
        models_dir=settings.models_dir
    ))
    await registry.set_active("litert_cli")

    engine = EngineManager(
        provider_registry=registry,
        hardware_info=hardware,
        max_vram_mb=settings.max_vram_usage_mb,
        model_unload_timeout=settings.model_unload_timeout_sec,
        model_absolute_lifetime=settings.model_absolute_lifetime_sec,
        anti_oom_threshold_mb=settings.anti_oom_threshold_mb,
    )

    # Register models defined in settings
    for role, cfg in settings.models.items():
        engine.register_model(
            model_id=role,
            model_path=cfg.get("file", ""),
            model_type=cfg.get("type", "general"),
            estimated_vram_mb=cfg.get("estimated_vram_mb", 1500),
        )

    # Start engine (runs the unload background loop)
    await engine.start()
    
    try:
        # 1. Cold Start Chat
        cold_chat = await run_inference(
            engine, "chat", "Why is the sky blue? Answer in 5 words.", "COLD START CHAT"
        )
        
        # 2. Warm Start Chat
        warm_chat = await run_inference(
            engine, "chat", "What is 2+2? Answer in 5 words.", "WARM START CHAT"
        )

        # 3. Model Swap (Chat -> Reasoning)
        swap_reasoning = await run_inference(
            engine, "reasoning", "Calculate 12*12. Answer in 5 words.", "MODEL SWAP CHAT -> REASONING"
        )

        # 4. Warm Start Reasoning
        warm_reasoning = await run_inference(
            engine, "reasoning", "Solve 10-4. Answer in 5 words.", "WARM START REASONING"
        )

        # 5. Check status endpoint response
        status = await engine.get_status()
        logger.info(f"Current Engine Status: {status}")

        # 6. Test Unload (Wait for 12 seconds to trigger the 10-second idle timeout)
        logger.info("Waiting 12 seconds to allow idle unload loop to run...")
        await asyncio.sleep(12)

        # 7. Check if model is unloaded
        status_after = await engine.get_status()
        logger.info(f"Engine Status after idle wait: {status_after}")

    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
