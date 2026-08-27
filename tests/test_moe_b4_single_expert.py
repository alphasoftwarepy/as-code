"""
AS-Core MoE Engine — Unit & Physical Tests for Bloque B4.1 (Single Expert Execution)
=====================================================================================
Valida en hardware real (NVIDIA GeForce GTX 1650 Ti):
1. Inferencia SwiGLU FFN de un único experto en GPU.
2. Equivalencia numérica estricta: Cosine Similarity >= 0.9999, Max Error < 1e-3.
3. Aislamiento estricto de memoria (no se cargan los otros 59 expertos).
4. Medición de tiempos de cómputo GPU y transferencia.
"""

import time
import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
from core.moe.single_expert_executor import SingleExpertExecutor

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
def single_executor(shared_registry, shared_cuda_driver, shared_cublas):
    executor = SingleExpertExecutor(
        registry=shared_registry,
        cuda_driver=shared_cuda_driver,
        cublas_backend=shared_cublas,
    )
    yield executor
    executor.release()


class TestSingleExpertExecution:
    """Pruebas físicas de ejecución de un único experto en GPU."""

    def test_b41_single_expert_numerical_exactness(self, single_executor):
        """[B4.1 Gate] Verifica la exactitud matemática contra referencia CPU:
        - Cosine Similarity >= 0.9999
        - Max Absolute Error < 1e-3
        """
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)

        # Ejecutar Layer 0, Expert 7
        res = single_executor.execute_single_expert(layer_id=0, expert_id=7, x=x, verify_against_cpu_ref=True)

        print(f"\n" + "=" * 70)
        print(" [B4.1 EMPIRICAL RESULT] SINGLE EXPERT GPU EXECUTION")
        print("=" * 70)
        print(f"  Experto: Layer 0, Expert ID 7")
        print(f"  • Cosine Similarity vs CPU Ref: {res.cosine_similarity_vs_ref:.7f} (Req >= 0.9999)")
        print(f"  • Max Absolute Error:           {res.max_absolute_error_vs_ref:.6e} (Req < 1e-3)")
        print(f"  • GPU Compute Time:             {res.compute_time_ms:.4f} ms")
        print(f"  • Weight Transfer Time:         {res.transfer_time_ms:.4f} ms")
        print(f"  • VRAM Usada por el Experto:    {res.vram_bytes_used / (1024*1024):.2f} MB")
        print("=" * 70)

        assert res.cosine_similarity_vs_ref >= 0.9999
        assert res.max_absolute_error_vs_ref < 1e-3
        assert res.is_numerically_exact is True
        assert res.output.shape == (1, 2048)

    def test_b41_mixed_quantization_layers(self, single_executor):
        """Verifica que el ejecutor opera con exactitud en capas con diferente cuantización (Q6_K vs Q4_K)."""
        np.random.seed(101)
        x = np.random.randn(1, 2048).astype(np.float32)

        # 1. Capa 0 (Q6_K down)
        res_l0 = single_executor.execute_single_expert(layer_id=0, expert_id=0, x=x, verify_against_cpu_ref=True)
        assert res_l0.cosine_similarity_vs_ref >= 0.9999
        assert res_l0.max_absolute_error_vs_ref < 1e-3

        # 2. Capa 3 (Q4_K down)
        res_l3 = single_executor.execute_single_expert(layer_id=3, expert_id=15, x=x, verify_against_cpu_ref=True)
        assert res_l3.cosine_similarity_vs_ref >= 0.9999
        assert res_l3.max_absolute_error_vs_ref < 1e-3

        # 3. Capa 12 (Capa profunda)
        res_l12 = single_executor.execute_single_expert(layer_id=12, expert_id=45, x=x, verify_against_cpu_ref=True)
        assert res_l12.cosine_similarity_vs_ref >= 0.9999
        assert res_l12.max_absolute_error_vs_ref < 1e-3

    def test_b41_isolation_and_vram_footprint(self, single_executor):
        """Demuestra que la ejecución de un experto NO carga el bloque monolítico de 360 MB."""
        np.random.seed(202)
        x = np.random.randn(1, 2048).astype(np.float32)

        res = single_executor.execute_single_expert(layer_id=1, expert_id=22, x=x, verify_against_cpu_ref=False)
        
        # El buffer de 1 experto en Float32 ocupa ~33 MB VRAM (en lugar de 360 MB monolíticos)
        assert res.vram_bytes_used < 40 * 1024 * 1024
        assert res.vram_bytes_used == (2 * 1408 * 2048 * 4) + (2048 * 1408 * 4)
