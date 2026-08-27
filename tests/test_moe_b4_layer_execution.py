"""
AS-Core MoE Engine — Unit & Physical Tests for Subfase B4.3.4 (Router + 4 Experts + Weighted Sum)
=================================================================================================
Valida en hardware real (NVIDIA GeForce GTX 1650 Ti):
1. Inferencia completa de una capa MoE: Router -> Top-4 -> Residency -> 4 FFN -> Weighted Sum en GPU.
2. Exactitud numérica estricta contra referencia CPU:
   - Cosine Similarity >= 0.9999
   - Max Absolute Error < 1e-3
   - Relative Error < 1e-4
3. Desglose desacoplado de latencias (Cold vs Warm).
4. Acumulación ponderada directa en GPU (cublasSaxpy sin copias intermedias Host).
5. Trazabilidad multi-capa y multi-token.
"""

import time
import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
from core.moe.layer_executor import MoELayerExecutor, MoELayerExecutionResult
from core.moe.residency_manager import ResidencyManager
from core.moe.router import RealRouter
from core.moe.vram_pool import VRAMExpertPool

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


@pytest.fixture
def layer_executor(shared_registry, shared_cuda_driver, shared_cublas):
    router = RealRouter(registry=shared_registry, k_active=4, cuda_driver=shared_cuda_driver, cublas_backend=shared_cublas)
    sample_exp = shared_registry.get_expert(0, 0)
    vram_pool = VRAMExpertPool(num_slots=4, slot_size_bytes=sample_exp.total_bytes, cuda_driver=shared_cuda_driver)
    res_mgr = ResidencyManager(registry=shared_registry, vram_pool=vram_pool)

    executor = MoELayerExecutor(
        registry=shared_registry,
        router=router,
        residency_manager=res_mgr,
        cuda_driver=shared_cuda_driver,
        cublas_backend=shared_cublas,
        k_active=4,
    )
    yield executor
    executor.release()
    vram_pool.release()
    router.release()


class TestMoELayerExecution:
    """Pruebas de la capa MoE completa (Subfase B4.3.4)."""

    def test_b434_cold_layer_execution_and_numerical_exactness(self, layer_executor):
        """[B4.3.4 Gate - Cold] Ejecuta la capa MoE en estado frío y valida tolerancias numéricas estrictas."""
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)

        res = layer_executor.forward_layer(layer_id=0, x=x, verify_against_cpu_ref=True)

        print(f"\n" + "=" * 80)
        print(" [B4.3.4 EMPIRICAL RESULT] COLD MOE LAYER FORWARD PASS (ROUTER + 4 EXPERTS + WEIGHTED SUM)")
        print("=" * 80)
        print(f"  Capa: Layer {res.layer_id} | Router Top-4 IDs: {res.expert_ids}")
        print(f"  Pesos Normalizados w_i: {[round(w, 6) for w in res.routing_weights]}")
        print(f"  • Cosine Similarity vs CPU Ref: {res.cosine_similarity_vs_ref:.7f} (Requisito >= 0.9999)")
        print(f"  • Max Absolute Error:           {res.max_absolute_error_vs_ref:.6e} (Requisito < 1e-3)")
        print(f"  • Relative Error:               {res.relative_error_vs_ref:.6e} (Requisito < 1e-4)")
        print(f"  • Router Latency:               {res.router_latency_ms:.3f} ms")
        print(f"  • Residency Lookup Latency:     {res.residency_lookup_latency_ms:.3f} ms")
        print(f"  • Promotion Latency (Cold):     {res.promotion_latency_ms:.3f} ms")
        print(f"  • 4 Experts Compute Latency:    {res.expert_compute_latency_ms:.3f} ms")
        print(f"  • Weighted Sum Latency (GPU):   {res.weighted_sum_latency_ms:.3f} ms")
        print(f"  • Total Layer Latency:          {res.total_layer_latency_ms:.3f} ms")
        print(f"  • Exactitud Numérica:           {res.is_exact}")
        print("=" * 80)

        assert res.is_exact is True
        assert res.cosine_similarity_vs_ref >= 0.9999
        assert res.max_absolute_error_vs_ref < 1e-3
        assert res.relative_error_vs_ref < 1e-4
        assert res.output.shape == (1, 2048)
        assert res.is_warm is False
        assert res.miss_count == 4

    def test_b434_warm_layer_execution_zero_promotion(self, layer_executor):
        """[B4.3.4 Gate - Warm] Mismo token en estado caliente: 4 HITs, 0 ms promoción, suma ponderada GPU."""
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)

        # Paso 1: Cold Run
        layer_executor.forward_layer(layer_id=0, x=x, verify_against_cpu_ref=False)

        # Paso 2: Warm Run
        res_warm = layer_executor.forward_layer(layer_id=0, x=x, verify_against_cpu_ref=True)

        print(f"\n" + "=" * 80)
        print(" [B4.3.4 EMPIRICAL RESULT] WARM MOE LAYER FORWARD PASS (0 TRANSFERENCIAS)")
        print("=" * 80)
        print(f"  • Hit Rate:                    {res_warm.hit_rate*100:.1f}% (4/4 HITs)")
        print(f"  • Promotion Latency (Warm):    {res_warm.promotion_latency_ms:.4f} ms (0 Bytes transferidos)")
        print(f"  • 4 Experts Compute Latency:   {res_warm.expert_compute_latency_ms:.3f} ms")
        print(f"  • Weighted Sum Latency (GPU):  {res_warm.weighted_sum_latency_ms:.3f} ms")
        print(f"  • Total Layer Latency (Warm):  {res_warm.total_layer_latency_ms:.3f} ms")
        print(f"  • Cosine Similarity vs Ref:    {res_warm.cosine_similarity_vs_ref:.7f}")
        print(f"  • Max Absolute Error:          {res_warm.max_absolute_error_vs_ref:.6e}")
        print("=" * 80)

        assert res_warm.is_warm is True
        assert res_warm.hit_count == 4
        assert res_warm.promotion_latency_ms == 0.0
        assert res_warm.is_exact is True

    def test_b434_multilayer_and_multitoken_execution(self, layer_executor):
        """Valida la inferencia completa en múltiples capas (0, 3, 11, 23) y tokens variables."""
        test_layers = [0, 3, 11, 23]
        np.random.seed(1234)

        for l_id in test_layers:
            for t_idx in range(3):
                x = np.random.randn(1, 2048).astype(np.float32)
                res = layer_executor.forward_layer(layer_id=l_id, x=x, verify_against_cpu_ref=True)
                assert res.is_exact is True, f"Fallo numérico en capa {l_id}, token {t_idx}"
                assert res.output.shape == (1, 2048)
