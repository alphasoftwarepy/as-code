"""
AS-Core MoE Engine — Unit & Integration Tests for Subfase B4.3.2 (Router + Expert Registry)
===========================================================================================
Valida la integración determinista entre RealRouter y ExpertRegistry:
1. Resolución automática Router -> Top-K IDs -> ExpertRegistry -> ExpertTensor.
2. Prueba de identidad estricta (100% coincidencia sin selección manual).
3. Validación multicapa (Layer 0, 5, 11, 18, 23).
4. Verificación de offsets continuos, tamaños y límites sin colisión entre expertos adyacentes.
5. Verificación de cuantización individual (Q4_K / Q6_K / Q5_K).
"""

import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
from core.moe.router import RealRouter, RoutedLayerDispatch

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


class TestRouterRegistryIntegration:
    """Pruebas de integración formal Router -> ExpertRegistry (B4.3.2)."""

    def test_b432_identity_and_resolution(self, real_router):
        """[B4.3.2 Gate] Prueba de identidad estricta: Router IDs == Registry ExpertTensor IDs."""
        np.random.seed(42)
        x = np.random.randn(1, 2048).astype(np.float32)

        dispatch = real_router.route_and_resolve(layer_id=0, x=x, use_gpu=True)

        print(f"\n" + "=" * 80)
        print(" [B4.3.2 EMPIRICAL RESULT] ROUTER -> REGISTRY RESOLUTION")
        print("=" * 80)
        print(f"  Capa: Layer {dispatch.layer_id} | Total Bytes Resueltos: {dispatch.total_bytes:,} ({dispatch.total_mb:.2f} MB)")
        print(f"  Router Decision Top-4 IDs: {dispatch.decision.top_k_ids}")
        for idx, re in enumerate(dispatch.routed_experts):
            exp = re.tensor
            print(
                f"  • Slot {idx}: Expert {re.expert_id} (Layer {re.layer_id}) | "
                f"Weight: {re.weight:.6f} | "
                f"Gate: offset={exp.gate_slice.offset_in_gguf}, size={exp.gate_slice.n_bytes} ({exp.gate_slice.quant_type}) | "
                f"Down: offset={exp.down_slice.offset_in_gguf}, size={exp.down_slice.n_bytes} ({exp.down_slice.quant_type})"
            )
        print(f"  • Latencia Total Resolución: {dispatch.resolution_latency_ms:.4f} ms")
        print(f"  • Verificación de Identidad e Integridad: {dispatch.is_valid}")
        print("=" * 80)

        assert dispatch.is_valid is True
        assert dispatch.expert_ids == dispatch.decision.top_k_ids
        assert len(dispatch.routed_experts) == 4

        for re in dispatch.routed_experts:
            assert re.layer_id == 0
            assert re.expert_id in dispatch.decision.top_k_ids
            assert re.tensor.is_complete is True
            assert re.tensor.gate_slice.n_bytes > 0
            assert re.tensor.up_slice.n_bytes > 0
            assert re.tensor.down_slice.n_bytes > 0

    def test_b432_multilayer_offset_boundaries(self, real_router, shared_registry):
        """Valida límites de offsets y no-solapamiento en capas estratégicas (0, 5, 11, 18, 23)."""
        test_layers = [0, 5, 11, 18, 23]
        np.random.seed(9876)

        for l_id in test_layers:
            x = np.random.randn(1, 2048).astype(np.float32)
            dispatch = real_router.route_and_resolve(layer_id=l_id, x=x, use_gpu=True)

            assert dispatch.is_valid is True
            assert dispatch.layer_id == l_id
            assert len(dispatch.routed_experts) == 4

            for re in dispatch.routed_experts:
                exp = re.tensor
                e_id = re.expert_id
                
                # Comprobación de límites contra B1
                direct_exp = shared_registry.get_expert(l_id, e_id)
                assert exp.gate_slice.offset_in_gguf == direct_exp.gate_slice.offset_in_gguf
                assert exp.up_slice.offset_in_gguf == direct_exp.up_slice.offset_in_gguf
                assert exp.down_slice.offset_in_gguf == direct_exp.down_slice.offset_in_gguf

                # Validar que no invade el offset del siguiente experto
                if e_id < shared_registry.profile.expert_count - 1:
                    next_exp = shared_registry.get_expert(l_id, e_id + 1)
                    assert exp.gate_slice.offset_in_gguf + exp.gate_slice.n_bytes == next_exp.gate_slice.offset_in_gguf

    def test_b432_quantization_metadata_integrity(self, real_router):
        """Comprueba que la cuantización mixta por capa se preserva fielmente."""
        x = np.random.randn(1, 2048).astype(np.float32)

        for layer_id in [0, 3, 5, 10, 20, 23]:
            dispatch = real_router.route_and_resolve(layer_id=layer_id, x=x, use_gpu=True)
            for re in dispatch.routed_experts:
                # Gate y Up siempre son Q4_K (12)
                assert re.tensor.gate_slice.quant_type in ("12", "GGML_TYPE_Q4_K")
                assert re.tensor.up_slice.quant_type in ("12", "GGML_TYPE_Q4_K")
                # Down es Q6_K (8) o Q5_K (6)
                assert re.tensor.down_slice.quant_type in ("8", "6", "GGML_TYPE_Q6_K", "GGML_TYPE_Q5_K")
                # Validación de tamaños exactos
                assert re.tensor.gate_slice.n_bytes == re.tensor.gate_slice.total_tensor_bytes // 60
                assert re.tensor.up_slice.n_bytes == re.tensor.up_slice.total_tensor_bytes // 60
                assert re.tensor.down_slice.n_bytes == re.tensor.down_slice.total_tensor_bytes // 60
