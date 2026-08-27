"""
AS-Core MoE Engine — Unit & Physical Tests for Bloque B4.2 (Four Experts Execution)
===================================================================================
Valida en hardware real (NVIDIA GeForce GTX 1650 Ti):
1. Alojamiento simultáneo de los 4 expertos activos (Routing Set) en VRAM.
2. Ejecución individual e independiente de cada uno de los 4 expertos en GPU.
3. Verificación de equivalencia numérica para los 4 expertos (Cosine Sim >= 0.9999, Max Err < 1e-3).
4. Verificación de huella de memoria: solo los 4 expertos activos residen en VRAM (el 93.3% restante permanece aislado).
"""

import time
import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
from core.moe.multi_expert_executor import MultiExpertExecutor

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
def multi_executor(shared_registry, shared_cuda_driver, shared_cublas):
    executor = MultiExpertExecutor(
        registry=shared_registry,
        k_active=4,
        cuda_driver=shared_cuda_driver,
        cublas_backend=shared_cublas,
    )
    yield executor
    executor.release()


class TestMultiExpertExecution:
    """Pruebas físicas del ejecutor de los 4 expertos activos del routing set."""

    def test_b42_four_experts_routing_set_execution(self, multi_executor):
        """[B4.2 Gate] Ejecuta 4 expertos reales de Qwen en GPU y valida exactitud numérica 4/4."""
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)
        active_ids = [3, 14, 27, 52]

        res = multi_executor.execute_routing_set(layer_id=0, expert_ids=active_ids, x=x, verify_against_cpu_ref=True)

        print(f"\n" + "=" * 75)
        print(" [B4.2 EMPIRICAL RESULT] FOUR EXPERTS GPU ROUTING SET EXECUTION")
        print("=" * 75)
        print(f"  Capa: Layer 0 | Expertos Activos IDs: {active_ids}")
        for idx, eid in enumerate(active_ids):
            print(f"  • Experto {eid} (Slot {idx}): Cosine Sim={res.cosine_similarities[idx]:.7f} | Max Err={res.max_absolute_errors[idx]:.6e} | Compute={res.individual_compute_times_ms[idx]:.3f}ms")
        print(f"  • Tiempo Total Cómputo GPU (4 Expertos): {res.total_gpu_compute_time_ms:.3f} ms")
        print(f"  • Tiempo de Transferencia a VRAM:        {res.total_transfer_time_ms:.3f} ms")
        print(f"  • VRAM Total Ocupada (4 Expertos):       {res.total_vram_bytes_allocated / (1024*1024):.2f} MB")
        print(f"  • Exactitud Numérica en los 4 Expertos:  {res.is_all_exact}")
        print("=" * 75)

        assert len(res.expert_outputs) == 4
        assert res.is_all_exact is True
        for sim in res.cosine_similarities:
            assert sim >= 0.9999
        for err in res.max_absolute_errors:
            assert err < 1e-3

    def test_b42_isolation_vs_monolithic_block(self, multi_executor):
        """Demuestra que la ejecución de los 4 expertos consume una fracción de la memoria
        y no carga los 56 expertos restantes de la capa.
        """
        np.random.seed(88)
        x = np.random.randn(1, 2048).astype(np.float32)
        active_ids = [0, 15, 30, 45]

        res = multi_executor.execute_routing_set(layer_id=1, expert_ids=active_ids, x=x, verify_against_cpu_ref=False)

        # 4 expertos en FP32 ocupan ~132 MB VRAM (en cuantizado original son 24 MB), muy inferior a los 360 MB de la capa completa
        assert res.total_vram_bytes_allocated < 150 * 1024 * 1024
        assert len(res.expert_outputs) == 4

    def test_b42_different_layers_routing_sets(self, multi_executor):
        """Verifica la ejecución de sets activos en capas intermedias y profundas."""
        np.random.seed(999)
        x = np.random.randn(1, 2048).astype(np.float32)

        # Capa 3 (Q4_K)
        res_l3 = multi_executor.execute_routing_set(layer_id=3, expert_ids=[1, 10, 20, 59], x=x, verify_against_cpu_ref=True)
        assert res_l3.is_all_exact is True

        # Capa 15 (Capa profunda)
        res_l15 = multi_executor.execute_routing_set(layer_id=15, expert_ids=[5, 12, 33, 44], x=x, verify_against_cpu_ref=True)
        assert res_l15.is_all_exact is True
