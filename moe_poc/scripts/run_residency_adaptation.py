"""
AS-Core MoE Engine — Residency Profile Learning & Cold Start Adaptation (B4.4.5 & B4.4.6)
========================================================================================
1. Genera y guarda el perfil de residencia aprendido en:
   moe_poc/data/residency_profile.json (B4.4.5)
2. Evalúa la adaptación de Cold Start mediante precarga en VRAM (B4.4.6):
   - Compara Cold Start (VRAM vacía) vs Profile Preloaded (Top-N precargados)
   - Mide el impacto en TTFT, velocidad del primer token y Hit Rate de los primeros 10 tokens.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, r"C:\as-code")

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.dynamic_residency_engine import DynamicResidencyEngine
from core.moe.expert_registry import ExpertRegistry
from core.moe.ram_warm_pool import RAMWarmPool
from core.moe.router import RealRouter

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
TRACE_PATH = r"C:\as-code\moe_poc\data\routing_trace.jsonl"
PROFILE_OUT = r"C:\as-code\moe_poc\data\residency_profile.json"
ADAPTATION_OUT = r"C:\as-code\moe_poc\data\cold_start_adaptation_results.json"


def build_learned_profile_from_trace(trace_path: str, registry: ExpertRegistry, slots_per_layer: int = 12) -> Dict[str, Any]:
    """Construye el perfil estadístico de residencia a partir de los 18,432 eventos del trace real (B4.4.5)."""
    from collections import Counter
    layer_counts = [Counter() for _ in range(24)]

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            l = ev["layer"]
            for eid in ev["experts"]:
                layer_counts[l][eid] += 1

    hot_experts_per_layer = {}
    layer_stats = {}

    for l in range(24):
        most_common = layer_counts[l].most_common()
        hot_eids = [eid for eid, _ in most_common[:slots_per_layer]]
        hot_experts_per_layer[str(l)] = hot_eids

        total_reqs = sum(layer_counts[l].values())
        layer_stats[str(l)] = {
            "total_requests": total_reqs,
            "top_experts": [
                {"expert_id": eid, "frequency": c, "percentage": round(c / total_reqs, 4)}
                for eid, c in most_common[:slots_per_layer]
            ]
        }

    profile = {
        "model_path": registry.profile.model_path,
        "architecture": registry.profile.architecture,
        "block_count": 24,
        "expert_count": 60,
        "expert_used_count": 4,
        "slots_per_layer": slots_per_layer,
        "total_trace_events": 18432,
        "hot_experts_per_layer": hot_experts_per_layer,
        "layer_stats": layer_stats,
    }

    Path(PROFILE_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_OUT, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"  [+] Perfil de residencia exportado exitosamente a: {PROFILE_OUT}")
    return profile


def evaluate_cold_start_vs_profile(
    profile_path: str,
    slots_per_layer: int = 12,
    num_eval_tokens: int = 10,
) -> Dict[str, Any]:
    """Evalúa comparativamente Cold Start vs Profile Preloaded en los primeros 10 tokens (B4.4.6)."""
    reg = ExpertRegistry(QWEN_PATH)
    driver = CUDADriver()
    cublas = CuBLASBackend(cuda_driver=driver)
    router = RealRouter(registry=reg, k_active=4, cuda_driver=driver, cublas_backend=cublas)

    # 1. EVALUACIÓN SCENARIO A: COLD START (VRAM Vacía)
    print("\n  [*] Evaluando Escenario A: COLD START (VRAM vacía, sin pre-carga)...")
    engine_cold = DynamicResidencyEngine(
        registry=reg,
        slots_per_layer=slots_per_layer,
        cuda_driver=driver,
    )

    np.random.seed(999)
    current = np.random.randn(2048).astype(np.float32)
    current /= np.linalg.norm(current)

    cold_first_token_ms = 0.0
    cold_token_times = []

    for t in range(num_eval_tokens):
        t0 = time.perf_counter()
        noise = np.random.randn(2048).astype(np.float32)
        noise /= np.linalg.norm(noise)
        current = 0.85 * current + np.sqrt(1.0 - 0.85**2) * noise
        x_vec = (current * 1.5).reshape(1, 2048)

        for l in range(24):
            dec = router.route_token(layer_id=l, x=x_vec, use_gpu=True)
            engine_cold.dispatch_layer(dec)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        cold_token_times.append(elapsed_ms)
        if t == 0:
            cold_first_token_ms = elapsed_ms

    metrics_cold = engine_cold.get_metrics()
    engine_cold.release()

    # 2. EVALUACIÓN SCENARIO B: PROFILE PRELOADED START (Top-N en VRAM)
    print("\n  [*] Evaluando Escenario B: PROFILE PRELOADED (Top-N expertos en VRAM)...")
    engine_warm = DynamicResidencyEngine(
        registry=reg,
        slots_per_layer=slots_per_layer,
        cuda_driver=driver,
    )

    t_preload_0 = time.perf_counter()
    preloaded = engine_warm.preload_from_profile(profile_path, top_n_per_layer=slots_per_layer)
    preload_time_ms = (time.perf_counter() - t_preload_0) * 1000.0
    print(f"      -> {preloaded} expertos precargados en VRAM en {preload_time_ms:.2f} ms.")

    engine_warm.reset_metrics()

    np.random.seed(999)
    current = np.random.randn(2048).astype(np.float32)
    current /= np.linalg.norm(current)

    warm_first_token_ms = 0.0
    warm_token_times = []

    for t in range(num_eval_tokens):
        t0 = time.perf_counter()
        noise = np.random.randn(2048).astype(np.float32)
        noise /= np.linalg.norm(noise)
        current = 0.85 * current + np.sqrt(1.0 - 0.85**2) * noise
        x_vec = (current * 1.5).reshape(1, 2048)

        for l in range(24):
            dec = router.route_token(layer_id=l, x=x_vec, use_gpu=True)
            engine_warm.dispatch_layer(dec)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        warm_token_times.append(elapsed_ms)
        if t == 0:
            warm_first_token_ms = elapsed_ms

    metrics_warm = engine_warm.get_metrics()
    engine_warm.release()
    router.release()
    cublas.destroy()
    driver.destroy()

    results = {
        "num_eval_tokens": num_eval_tokens,
        "slots_per_layer": slots_per_layer,
        "cold_start": {
            "first_token_latency_ms": round(cold_first_token_ms, 2),
            "first_token_tok_s": round(1000.0 / cold_first_token_ms, 2) if cold_first_token_ms > 0 else 0.0,
            "avg_latency_10_tokens_ms": round(float(np.mean(cold_token_times)), 2),
            "avg_tok_s_10_tokens": round(1000.0 / float(np.mean(cold_token_times)), 2),
            "hit_rate_10_tokens_pct": round(metrics_cold["hit_rate"] * 100, 2),
            "promotions_10_tokens": metrics_cold["promotions"],
            "bytes_transferred_mb": metrics_cold["bytes_h2d_mb"],
        },
        "profile_preloaded_start": {
            "preload_time_ms": round(preload_time_ms, 2),
            "preloaded_experts_count": preloaded,
            "first_token_latency_ms": round(warm_first_token_ms, 2),
            "first_token_tok_s": round(1000.0 / warm_first_token_ms, 2) if warm_first_token_ms > 0 else 0.0,
            "avg_latency_10_tokens_ms": round(float(np.mean(warm_token_times)), 2),
            "avg_tok_s_10_tokens": round(1000.0 / float(np.mean(warm_token_times)), 2),
            "hit_rate_10_tokens_pct": round(metrics_warm["hit_rate"] * 100, 2),
            "promotions_10_tokens": metrics_warm["promotions"],
            "bytes_transferred_mb": metrics_warm["bytes_h2d_mb"],
        },
        "improvements": {
            "first_token_speedup_x": round(cold_first_token_ms / warm_first_token_ms, 2) if warm_first_token_ms > 0 else 1.0,
            "hit_rate_increase_percentage_points": round((metrics_warm["hit_rate"] - metrics_cold["hit_rate"]) * 100, 2),
            "promotions_reduced_pct": round((1.0 - (metrics_warm["promotions"] / metrics_cold["promotions"])) * 100, 2) if metrics_cold["promotions"] > 0 else 0.0,
            "pcie_bandwidth_saved_mb": round(metrics_cold["bytes_h2d_mb"] - metrics_warm["bytes_h2d_mb"], 2),
        }
    }

    with open(ADAPTATION_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  [+] Resultados de adaptación guardados en: {ADAPTATION_OUT}")
    return results


def main():
    print("=" * 80)
    print(" [AS-CORE B4.4.5 & B4.4.6] PERFIL DE RESIDENCIA & ADAPTACIÓN DE COLD START")
    print("=" * 80)

    reg = ExpertRegistry(QWEN_PATH)

    # 1. B4.4.5 — Generar residency_profile.json
    print("\n[FASE B4.4.5] Generando perfil de residencia aprendido desde el trace...")
    profile = build_learned_profile_from_trace(TRACE_PATH, reg, slots_per_layer=12)

    # 2. B4.4.6 — Evaluar adaptación de Cold Start
    print("\n[FASE B4.4.6] Evaluando impacto de la precarga de perfil en Cold Start...")
    adapt_results = evaluate_cold_start_vs_profile(PROFILE_OUT, slots_per_layer=12, num_eval_tokens=10)

    # Imprimir resumen de comparación
    cold = adapt_results["cold_start"]
    warm = adapt_results["profile_preloaded_start"]
    imp = adapt_results["improvements"]

    print("\n" + "=" * 80)
    print(" [RESUMEN DE ADAPTACIÓN B4.4.6] COLD START vs PROFILE PRELOADED (10 TOKENS)")
    print("=" * 80)
    print(f" Métrica                          | Cold Start       | Profile Preloaded | Mejora")
    print("-" * 80)
    print(f" Latencia 1er Token (TTFT)        | {cold['first_token_latency_ms']:>8.1f} ms    | {warm['first_token_latency_ms']:>8.1f} ms      | {imp['first_token_speedup_x']}x más rápido")
    print(f" Tok/s 1er Token                  | {cold['first_token_tok_s']:>8.2f} tok/s   | {warm['first_token_tok_s']:>8.2f} tok/s     | +{warm['first_token_tok_s'] - cold['first_token_tok_s']:.2f} tok/s")
    print(f" Hit Rate (Primeros 10 tokens)    | {cold['hit_rate_10_tokens_pct']:>8.1f} %       | {warm['hit_rate_10_tokens_pct']:>8.1f} %        | +{imp['hit_rate_increase_percentage_points']:.1f}%")
    print(f" Promociones PCIe (10 tokens)     | {cold['promotions_10_tokens']:>8} fallos   | {warm['promotions_10_tokens']:>8} fallos     | -{imp['promotions_reduced_pct']:.1f}% fallos")
    print(f" Tráfico PCIe Transferido         | {cold['bytes_transferred_mb']:>8.1f} MB     | {warm['bytes_transferred_mb']:>8.1f} MB       | -{imp['pcie_bandwidth_saved_mb']:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
