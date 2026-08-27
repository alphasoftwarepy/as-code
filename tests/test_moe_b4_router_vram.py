"""
AS-Core MoE Engine — Unit & Physical Tests for Subfase B4.3.3 (Router + VRAM Expert Pool)
=========================================================================================
Valida la conexión física dinámica entre RealRouter, ExpertRegistry y VRAMExpertPool:
1. Resolución dinámica: Hidden state -> Router -> Registry -> VRAM slots.
2. Primer ciclo (Cold): 4 MISSes -> Promoción física a VRAM.
3. Segundo ciclo (Warm): 4 HITs -> 0 bytes transferidos, 0 ms latencia de promoción.
4. Reutilización de slots (Slot Reuse): Reemplazo in-place sin cuMemFree/cuMemAlloc.
5. Inferencia multi-token dinámica y verificación estricta de seguridad de memoria.
"""

import time
import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
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


@pytest.fixture
def clean_residency_manager(shared_registry, shared_cuda_driver):
    sample_exp = shared_registry.get_expert(0, 0)
    pool = VRAMExpertPool(
        num_slots=4,
        slot_size_bytes=sample_exp.total_bytes,
        cuda_driver=shared_cuda_driver,
    )
    manager = ResidencyManager(registry=shared_registry, vram_pool=pool)
    yield manager
    pool.release()


class TestRouterVRAMIntegration:
    """Pruebas de la integración Router -> Registry -> VRAM Pool (B4.3.3)."""

    def test_b433_first_cycle_cold_miss_and_promotion(self, real_router, clean_residency_manager):
        """[B4.3.3.5] Primer ciclo: 4 expertos decididos dinámicamente resultan en 4 MISSes y promoción exitosa."""
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)

        # 1. Router decide dinámicamente los Top-4
        decision = real_router.route_token(layer_id=0, x=x, use_gpu=True)
        assert len(decision.top_k_ids) == 4

        # 2. Despachar al ResidencyManager
        dispatch = clean_residency_manager.dispatch_routing(decision)

        print(f"\n" + "=" * 80)
        print(" [B4.3.3.5 EMPIRICAL RESULT] CYCLE 1: COLD START PROMOTION")
        print("=" * 80)
        print(f"  Capa: {dispatch.layer_id} | Router Top-4 IDs: {dispatch.expert_ids}")
        for rd in dispatch.residency_decisions:
            print(
                f"  • Expert {rd.expert_id} -> Slot {rd.slot_id} | "
                f"Status: {'HIT' if rd.is_hit else 'MISS'} | "
                f"Promotion Time: {rd.promotion_latency_ms:.3f} ms | "
                f"Ptr: {hex(rd.device_ptr)}"
            )
        print(f"  • Hits: {dispatch.hit_count} | Misses: {dispatch.miss_count} | Hit Rate: {dispatch.hit_rate*100:.1f}%")
        print(f"  • Latencia Total Promoción: {dispatch.total_promotion_latency_ms:.3f} ms")
        print(f"  • VRAM Asignada: {dispatch.total_vram_allocated_bytes / (1024*1024):.2f} MB")
        print("=" * 80)

        assert dispatch.miss_count == 4
        assert dispatch.hit_count == 0
        assert dispatch.hit_rate == 0.0
        assert len(dispatch.device_ptrs) == 4
        assert all(ptr != 0 for ptr in dispatch.device_ptrs)

    def test_b433_second_cycle_warm_hit_path(self, real_router, clean_residency_manager):
        """[B4.3.3.6] Segundo ciclo: Mismo routing produce 4 HITs, 0 transferencias y 0 ms promoción."""
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)

        decision = real_router.route_token(layer_id=0, x=x, use_gpu=True)

        # Ciclo 1 (Warm up)
        clean_residency_manager.dispatch_routing(decision)

        # Ciclo 2 (Test HIT)
        t_hit_0 = time.perf_counter()
        dispatch_warm = clean_residency_manager.dispatch_routing(decision)
        t_hit_total = (time.perf_counter() - t_hit_0) * 1000.0

        print(f"\n" + "=" * 80)
        print(" [B4.3.3.6 EMPIRICAL RESULT] CYCLE 2: WARM HIT RESIDENCY")
        print("=" * 80)
        print(f"  Capa: {dispatch_warm.layer_id} | Router Top-4 IDs: {dispatch_warm.expert_ids}")
        for rd in dispatch_warm.residency_decisions:
            print(
                f"  • Expert {rd.expert_id} -> Slot {rd.slot_id} | "
                f"Status: {'HIT' if rd.is_hit else 'MISS'} | "
                f"Promotion Time: {rd.promotion_latency_ms:.3f} ms (0 Bytes transferidos)"
            )
        print(f"  • Hits: {dispatch_warm.hit_count} | Misses: {dispatch_warm.miss_count} | Hit Rate: {dispatch_warm.hit_rate*100:.1f}%")
        print(f"  • Latencia de Promoción: {dispatch_warm.total_promotion_latency_ms:.4f} ms")
        print(f"  • Latencia Total de Resolución HIT: {t_hit_total:.4f} ms")
        print("=" * 80)

        assert dispatch_warm.hit_count == 4
        assert dispatch_warm.miss_count == 0
        assert dispatch_warm.hit_rate == 1.0
        assert dispatch_warm.total_promotion_latency_ms == 0.0

    def test_b433_slot_reuse_and_eviction(self, real_router, clean_residency_manager):
        """[B4.3.3.7] Reutilización forzada de slots: Cambio de routing set reutiliza los 4 slots en VRAM."""
        # 1. Conjunto A (Seed 42)
        np.random.seed(42)
        x_a = np.random.randn(1, 2048).astype(np.float32)
        decision_a = real_router.route_token(layer_id=0, x=x_a, use_gpu=True)
        dispatch_a = clean_residency_manager.dispatch_routing(decision_a)
        assert dispatch_a.miss_count == 4

        # 2. Conjunto B (Seed 999 produce diferentes expertos)
        np.random.seed(999)
        x_b = np.random.randn(1, 2048).astype(np.float32)
        decision_b = real_router.route_token(layer_id=0, x=x_b, use_gpu=True)
        dispatch_b = clean_residency_manager.dispatch_routing(decision_b)

        print(f"\n" + "=" * 80)
        print(" [B4.3.3.7 EMPIRICAL RESULT] SLOT REUSE AND EVICTION")
        print("=" * 80)
        print(f"  Set A Expertos: {dispatch_a.expert_ids}")
        print(f"  Set B Expertos: {dispatch_b.expert_ids}")
        for rd in dispatch_b.residency_decisions:
            print(f"  • Slot {rd.slot_id}: Promoted Exp {rd.expert_id} | Evicted: {rd.evicted_expert}")
        print(f"  • Total Evictions Registradas: {clean_residency_manager.cumulative_evictions}")
        print("=" * 80)

        assert len(dispatch_b.residency_decisions) == 4
        assert clean_residency_manager.cumulative_evictions > 0

    def test_b433_four_dynamic_experts_multitoken_trace(self, real_router, clean_residency_manager):
        """[B4.3.3.8] Simulación de secuencia de 10 tokens con registro de métricas dinámicas."""
        np.random.seed(777)
        total_tokens = 10

        for t_idx in range(total_tokens):
            x = np.random.randn(1, 2048).astype(np.float32)
            layer_id = t_idx % 3 # Capas 0, 1, 2
            decision = real_router.route_token(layer_id=layer_id, x=x, use_gpu=True)
            clean_residency_manager.dispatch_routing(decision)

        metrics = clean_residency_manager.get_metrics()
        print(f"\n[B4.3.3 Multi-token Metrics]: {metrics}")

        assert metrics["total_requests"] == total_tokens * 4
        assert metrics["vram_slots_occupied"] <= 4 # No excede el límite del pool
        assert metrics["vram_allocated_mb"] == clean_residency_manager.vram_pool.allocated_mb
