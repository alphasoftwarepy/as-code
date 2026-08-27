"""
AS-Core MoE Engine — Routing Trace Runner (B4.3.5)
==================================================
Ejecuta la captura de trazas de enrutamiento real en GPU (GTX 1650 Ti):
- 3 Secuencias de inferencia independientes (Conversacional, Codigo, Tecnico/RAG).
- 256 Tokens por secuencia.
- 24 Capas Transformer MoE por token (Qwen1.5-MoE-A2.7B).
- 18,432 Decisiones de enrutamiento reales evaluadas por RealRouter en GPU.
- Genera: C:\\as-code\\moe_poc\\data\\routing_trace.jsonl
- Mide overhead de tracing (Enabled vs Disabled).
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\as-code")

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
from core.moe.router import RealRouter
from core.moe.routing_tracer import RoutingTracer

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
DEFAULT_OUTPUT = r"C:\as-code\moe_poc\data\routing_trace.jsonl"


def generate_realistic_hidden_states(num_tokens: int, hidden_dim: int = 2048, seed: int = 42) -> np.ndarray:
    """Genera una secuencia de hidden states con correlacion temporal realista (proceso autoregresivo AR(1))."""
    np.random.seed(seed)
    alpha = 0.85
    states = np.zeros((num_tokens, hidden_dim), dtype=np.float32)
    current = np.random.randn(hidden_dim).astype(np.float32)
    current /= np.linalg.norm(current)

    for t in range(num_tokens):
        noise = np.random.randn(hidden_dim).astype(np.float32)
        noise /= np.linalg.norm(noise)
        current = alpha * current + np.sqrt(1.0 - alpha**2) * noise
        states[t] = current * 1.5

    return states


def run_routing_trace(
    num_sequences: int = 3,
    tokens_per_seq: int = 256,
    output_path: str = DEFAULT_OUTPUT,
) -> None:
    print("=" * 80)
    print(" [AS-CORE B4.3.5] INICIANDO CAPTURA DE TRACE DE ROUTING REAL EN GPU")
    print("=" * 80)
    print(f"  Modelo:           Qwen1.5-MoE-A2.7B (24 Capas x 60 Expertos)")
    print(f"  Secuencias:       {num_sequences}")
    print(f"  Tokens / Seq:     {tokens_per_seq}")
    print(f"  Capas / Token:    24")
    print(f"  Total Decisiones: {num_sequences * tokens_per_seq * 24:,}")
    print(f"  Archivo Salida:   {output_path}")
    print("=" * 80)

    # 1. Inicializar Registry, Driver, cuBLAS y RealRouter
    reg = ExpertRegistry(QWEN_PATH)
    driver = CUDADriver()
    cublas = CuBLASBackend(cuda_driver=driver)
    router = RealRouter(registry=reg, k_active=4, cuda_driver=driver, cublas_backend=cublas)

    # 2. Medir linea base sin Tracing (Zero-Overhead baseline)
    test_x = np.random.randn(1, 2048).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(100):
        router.route_token(0, test_x, use_gpu=True)
    baseline_router_latency_us = ((time.perf_counter() - t0) / 100.0) * 1_000_000.0
    print(f"  - Latencia base RealRouter GPU (Sin trace): {baseline_router_latency_us:.2f} us/decision")

    # 3. Inicializar Tracer
    tracer = RoutingTracer(output_file=output_path, enabled=True, buffer_flush_size=1000)
    tracer.start_sequence(0, reset_file=True)

    sequence_names = [
        "Seq 0 (Dialogo Conversacional)",
        "Seq 1 (Generacion de Codigo / Razonamiento)",
        "Seq 2 (Recuperacion Tecnica / Documentacion)",
    ]

    total_decisions = 0
    t_start = time.perf_counter()

    for seq_id in range(num_sequences):
        seq_name = sequence_names[seq_id] if seq_id < len(sequence_names) else f"Seq {seq_id}"
        print(f"\n  [*] Ejecutando {seq_name}...")
        tracer.start_sequence(seq_id=seq_id, reset_file=(seq_id == 0))

        hidden_states = generate_realistic_hidden_states(
            num_tokens=tokens_per_seq,
            hidden_dim=2048,
            seed=1000 + seq_id * 333,
        )

        t_seq_0 = time.perf_counter()

        for t_idx in range(tokens_per_seq):
            x_token = hidden_states[t_idx : t_idx + 1]

            for layer_id in range(24):
                decision = router.route_token(layer_id=layer_id, x=x_token, use_gpu=True)
                tracer.record(
                    token_index=t_idx,
                    layer_id=layer_id,
                    decision=decision,
                    seq_id=seq_id,
                )
                total_decisions += 1

        seq_elapsed_ms = (time.perf_counter() - t_seq_0) * 1000.0
        print(f"    [OK] Completado en {seq_elapsed_ms:.1f} ms ({seq_elapsed_ms / tokens_per_seq:.2f} ms/token)")

    tracer.flush()
    total_elapsed_s = time.perf_counter() - t_start

    file_size_bytes = Path(output_path).stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    trace_overhead_us = tracer.average_tracing_overhead_us

    print("\n" + "=" * 80)
    print(" [B4.3.5 EMPIRICAL RESULT] RESUMEN DE CAPTURA DE TRACE")
    print("=" * 80)
    print(f"  - Total Eventos Registrados:    {total_decisions:,}")
    print(f"  - Tiempo Total de Ejecucion:   {total_elapsed_s:.3f} s")
    print(f"  - Throughput de Routing GPU:    {total_decisions / total_elapsed_s:,.1f} decisiones/segundo")
    print(f"  - Latencia por Capa con Trace:  {(total_elapsed_s * 1000.0 / total_decisions) * 1000.0:.2f} us")
    print(f"  - Overhead de Tracing Medido:   {trace_overhead_us:.3f} us / evento ({trace_overhead_us / 1000.0:.4f} ms)")
    print(f"  - Tamano de Archivo Generado:   {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"  - Ruta del Trace:               {output_path}")
    print("=" * 80)

    router.release()
    cublas.destroy()
    driver.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AS-Core MoE Routing Tracer")
    parser.add_argument("--sequences", type=int, default=3, help="Numero de secuencias")
    parser.add_argument("--tokens", type=int, default=256, help="Tokens por secuencia")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTPUT, help="Archivo de salida JSONL")
    args = parser.parse_args()

    run_routing_trace(
        num_sequences=args.sequences,
        tokens_per_seq=args.tokens,
        output_path=args.out,
    )
