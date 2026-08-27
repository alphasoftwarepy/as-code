"""
AS-Core MoE Engine — Unit & Physical Tests for Subfase B4.3.1 (Isolated Real Router)
====================================================================================
Valida en hardware real (NVIDIA GeForce GTX 1650 Ti):
1. Proyección lineal y Softmax estable sobre las 24 matrices reales de router de Qwen.
2. 100% de coincidencia de Expert IDs entre GPU y CPU de referencia.
3. Error relativo de Router Weights < 1e-5.
4. Normalización estricta: sum(weights) == 1.0.
"""

import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
from core.moe.router import RealRouter, RoutingDecision

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"


@pytest.fixture(scope="module")
def shared_registry():
    return ExpertRegistry(QWEN_PATH)


@pytest.fixture(scope="module")
def shared_cuda_driver():
    driver = CUDADriver()
    yield driver
    driver.destroy()


@pytest.fixture(scope="module")
def shared_cublas(shared_cuda_driver):
    backend = CuBLASBackend(cuda_driver=shared_cuda_driver)
    yield backend
    backend.destroy()


@pytest.fixture(scope="module")
def real_router(shared_registry, shared_cuda_driver, shared_cublas):
    router = RealRouter(
        registry=shared_registry,
        k_active=4,
        cuda_driver=shared_cuda_driver,
        cublas_backend=shared_cublas,
    )
    yield router
    router.release()


class TestIsolatedRealRouter:
    """Pruebas del Router real aislado (Subfase B4.3.1)."""

    def test_b431_topk_100pct_match_and_weights_tolerance(self, real_router):
        """[B4.3.1 Gate] Verifica 100% coincidencia de Top-K IDs y error relativo < 1e-5 en Layer 0."""
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)

        # 1. Referencia CPU
        ref_ids, ref_weights, ref_probs = real_router.route_cpu_reference(layer_id=0, x=x)

        # 2. Ejecución GPU
        decision = real_router.route_token(layer_id=0, x=x, use_gpu=True)

        # 3. Métricas
        ids_match = (decision.top_k_ids == ref_ids)
        max_abs_err = np.max(np.abs(np.array(decision.normalized_weights) - np.array(ref_weights)))
        max_rel_err = np.max(np.abs(np.array(decision.normalized_weights) - np.array(ref_weights)) / np.array(ref_weights))
        weights_sum = sum(decision.normalized_weights)

        print(f"\n" + "=" * 75)
        print(" [B4.3.1 EMPIRICAL RESULT] ISOLATED REAL ROUTER EVALUATION")
        print("=" * 75)
        print(f"  Capa: Layer 0 (60 Expertos, Top-4)")
        print(f"  • CPU Reference Top-4 IDs: {ref_ids}")
        print(f"  • GPU Evaluated Top-4 IDs: {decision.top_k_ids}")
        print(f"  • Coincidencia de IDs:     {ids_match} (Requisito: 100% idénticos)")
        print(f"  • Max Absolute Error:      {max_abs_err:.6e}")
        print(f"  • Max Relative Error:      {max_rel_err:.6e} (Requisito < 1e-5)")
        print(f"  • Suma de Pesos Top-4:     {weights_sum:.7f} (Requisito = 1.0)")
        print(f"  • Latencia de Enrutamiento: {decision.router_latency_ms:.4f} ms")
        print("=" * 75)

        assert ids_match is True
        assert max_rel_err < 1e-5
        assert abs(weights_sum - 1.0) < 1e-5

    def test_b431_multi_layer_comprehensive_validation(self, real_router):
        """Valida que el Router coincide al 100% en múltiples capas y con 20 vectores aleatorios diferentes."""
        np.random.seed(12345)
        test_layers = [0, 5, 11, 18, 23]

        for layer_id in test_layers:
            for trial in range(5):
                x = np.random.randn(1, 2048).astype(np.float32)

                ref_ids, ref_weights, _ = real_router.route_cpu_reference(layer_id=layer_id, x=x)
                decision = real_router.route_token(layer_id=layer_id, x=x, use_gpu=True)

                assert decision.top_k_ids == ref_ids, f"Fallo en Capa {layer_id}, trial {trial}: {decision.top_k_ids} vs {ref_ids}"
                max_rel_err = np.max(np.abs(np.array(decision.normalized_weights) - np.array(ref_weights)) / np.array(ref_weights))
                assert max_rel_err < 1e-5
                assert abs(sum(decision.normalized_weights) - 1.0) < 1e-5

    def test_b431_router_latency(self, real_router):
        """Mide la latencia de evaluación del router en GPU (debe ser sub-milisegundo)."""
        x = np.random.randn(1, 2048).astype(np.float32)
        latencies = []

        for _ in range(20):
            decision = real_router.route_token(layer_id=0, x=x, use_gpu=True)
            latencies.append(decision.router_latency_ms)

        avg_latency_ms = sum(latencies[5:]) / len(latencies[5:])
        print(f"\n[Router Microbenchmark]: Latencia promedio GPU = {avg_latency_ms:.4f} ms")
        assert avg_latency_ms < 1.0 # O(1) sub-milisegundo
