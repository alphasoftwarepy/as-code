"""
AS Code — Benchmark Runner

Automated benchmarks for:
- tokens/sec
- VRAM usage
- RAM usage
- cold start / warm start latency
- model swap speed
- first token latency
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

from core.hardware import detect_hardware, get_ram_available_mb, get_vram_free_mb

logger = logging.getLogger("as-code.benchmarks")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_id: str = ""
    provider: str = ""
    hardware_tier: str = ""

    # Latency
    cold_start_ms: float = 0.0
    warm_start_ms: float = 0.0
    first_token_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Throughput
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0

    # Resources
    ram_before_mb: int = 0
    ram_after_mb: int = 0
    vram_before_mb: int = 0
    vram_after_mb: int = 0
    peak_vram_mb: int = 0

    # Metadata
    prompt_length: int = 0
    max_tokens: int = 0
    temperature: float = 0.0
    quantization: str = ""
    timestamp: float = field(default_factory=time.time)


async def run_benchmark(
    engine,
    model_id: str,
    prompt: str = "Write a Python function that sorts a list using quicksort.",
    max_tokens: int = 256,
    runs: int = 3,
) -> list[BenchmarkResult]:
    """Run benchmark suite for a model.

    Performs:
    1. Cold start (first load)
    2. Warm inference (subsequent calls)
    3. Resource measurement before/after
    """
    from providers.base import InferenceRequest

    results = []

    for i in range(runs):
        result = BenchmarkResult(
            model_id=model_id,
            prompt_length=len(prompt),
            max_tokens=max_tokens,
        )

        # Measure resources before
        result.ram_before_mb = get_ram_available_mb()
        result.vram_before_mb = get_vram_free_mb()

        # Build request
        request = InferenceRequest(
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=0.7,
            request_id=f"bench_{i}",
        )

        # Time inference
        start = time.perf_counter()
        first_token_time = None
        tokens = 0

        async for chunk in engine.generate_stream(request):
            if first_token_time is None and chunk.text:
                first_token_time = time.perf_counter()
            tokens += len(chunk.text.split()) if chunk.text else 0

        end = time.perf_counter()

        # Record results
        total_ms = (end - start) * 1000
        result.total_latency_ms = round(total_ms, 1)
        result.tokens_generated = tokens
        result.tokens_per_sec = round(
            tokens / ((end - start) or 0.001), 1
        )

        if first_token_time:
            result.first_token_ms = round(
                (first_token_time - start) * 1000, 1
            )

        if i == 0:
            result.cold_start_ms = result.total_latency_ms
        else:
            result.warm_start_ms = result.total_latency_ms

        # Resources after
        result.ram_after_mb = get_ram_available_mb()
        result.vram_after_mb = get_vram_free_mb()

        results.append(result)
        logger.info(
            f"Benchmark run {i+1}/{runs}: "
            f"{result.tokens_per_sec} tok/s, "
            f"{result.total_latency_ms}ms total, "
            f"{result.first_token_ms}ms first token"
        )

    return results


def export_results(
    results: list[BenchmarkResult],
    output_path: str = "benchmarks/results.json",
) -> None:
    """Export benchmark results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Benchmark results exported to {output_path}")
