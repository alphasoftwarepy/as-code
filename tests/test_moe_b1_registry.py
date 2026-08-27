"""
AS-Core MoE Engine — Unit Tests for Bloque B1 (ModelProfile & ExpertRegistry)
=============================================================================
Valida el descubrimiento dinámico, indexación precisa y agnosticismo sobre:
- Qwen1.5-MoE-A2.7B (24 capas, 60 expertos, 4 activos)
- OLMoE-1B-7B (16 capas, 64 expertos, 8 activos)
"""

import os
from pathlib import Path
import pytest

from core.moe.expert_tensor import ExpertTensor, ExpertTensorSlice, ResidencyTier, TensorRole
from core.moe.model_profile import ModelProfile
from core.moe.expert_registry import ExpertRegistry

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
OLMOE_PATH = r"C:\as-code\moe_poc\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"

# Módulos en memoria cacheados para velocidad de ejecución de tests
_cached_qwen_registry = None
_cached_olmoe_registry = None

@pytest.fixture(scope="module")
def qwen_registry():
    global _cached_qwen_registry
    if _cached_qwen_registry is None:
        _cached_qwen_registry = ExpertRegistry(QWEN_PATH)
    return _cached_qwen_registry

@pytest.fixture(scope="module")
def olmoe_registry():
    global _cached_olmoe_registry
    if _cached_olmoe_registry is None:
        _cached_olmoe_registry = ExpertRegistry(OLMOE_PATH)
    return _cached_olmoe_registry


class TestModelProfile:
    """Pruebas del analizador dinámico de metadatos de modelos GGUF."""

    def test_qwen_profile_discovery(self, qwen_registry):
        """Verifica que Qwen1.5-MoE se profilea correctamente sin hardcoding."""
        profile = qwen_registry.profile

        assert profile.is_moe is True
        assert profile.architecture == "qwen2moe"
        assert profile.block_count == 24
        assert profile.expert_count == 60
        assert profile.expert_used_count == 4
        assert profile.total_experts_in_model == 1440
        assert profile.embedding_length == 2048
        assert profile.expert_feed_forward_length == 1408
        assert profile.total_file_size_gb == pytest.approx(8.84, rel=0.05)
        assert profile.dense_memory_mb > 1000.0
        assert profile.moe_memory_mb > 7000.0
        assert profile.single_expert_mb == pytest.approx(6.02, rel=0.10)
        assert profile.active_experts_layer_mb > 20.0

    def test_olmoe_profile_discovery(self, olmoe_registry):
        """Verifica que OLMoE-1B-7B se profilea correctamente con arquitectura diferente."""
        profile = olmoe_registry.profile

        assert profile.is_moe is True
        assert profile.architecture == "olmoe"
        assert profile.block_count == 16
        assert profile.expert_count == 64
        assert profile.expert_used_count == 8
        assert profile.total_experts_in_model == 1024
        assert profile.embedding_length == 2048
        assert profile.total_file_size_gb == pytest.approx(3.92, rel=0.05)
        assert profile.single_expert_mb > 0

    def test_hotset_capacity_calculation(self, qwen_registry):
        """Verifica la fórmula de capacidad de HotSet en VRAM."""
        profile = qwen_registry.profile
        
        # Con 4096 MB VRAM, 350 MB KV cache:
        hotset_total = profile.calculate_hotset_capacity(available_vram_mb=4096.0, kv_cache_mb=350.0)
        hotset_per_layer = profile.calculate_hotset_per_layer(available_vram_mb=4096.0, kv_cache_mb=350.0)

        assert hotset_total > 350
        assert hotset_per_layer >= 15
        assert hotset_per_layer <= profile.expert_count

    def test_nonexistent_file_raises_error(self):
        """Verifica que un archivo inexistente lanza FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ModelProfile.from_gguf("C:/path/does/not/exist.gguf")


class TestExpertRegistry:
    """Pruebas del registro e indexador de expertos individuales."""

    def test_qwen_registry_structure(self, qwen_registry):
        """Valida que todas las capas y expertos de Qwen estén correctamente indexados."""
        assert qwen_registry.total_layers == 24
        assert qwen_registry.total_experts_count == 1440
        assert qwen_registry.validate_integrity() is True

        # Inspeccionar un experto específico en capa 0
        exp0 = qwen_registry.get_expert(layer_id=0, expert_id=7)
        assert isinstance(exp0, ExpertTensor)
        assert exp0.layer_id == 0
        assert exp0.expert_id == 7
        assert exp0.is_complete is True
        assert exp0.gate_slice is not None
        assert exp0.up_slice is not None
        assert exp0.down_slice is not None
        assert exp0.total_bytes == 6307840
        assert exp0.total_mb == pytest.approx(6.02, rel=0.02)

    def test_olmoe_registry_structure(self, olmoe_registry):
        """Valida que todas las capas y expertos de OLMoE estén correctamente indexados."""
        assert olmoe_registry.total_layers == 16
        assert olmoe_registry.total_experts_count == 1024
        assert olmoe_registry.validate_integrity() is True

        exp = olmoe_registry.get_expert(layer_id=15, expert_id=63)
        assert exp.is_complete is True
        assert exp.layer_id == 15
        assert exp.expert_id == 63

    def test_tensor_slice_offsets_validity(self, qwen_registry):
        """Verifica que los offsets de los slices correspondan a posiciones válidas dentro del archivo."""
        file_size = Path(QWEN_PATH).stat().st_size

        for layer_id in [0, 11, 23]:
            for exp_id in [0, 29, 59]:
                exp = qwen_registry.get_expert(layer_id, exp_id)
                for s in exp.get_slices():
                    assert s.offset_in_gguf > 0
                    assert s.offset_in_gguf + s.n_bytes <= file_size
                    assert s.n_bytes > 0
                    assert s.total_experts == 60

    def test_routing_set_calculation(self, qwen_registry):
        """Valida el cálculo de memoria para un set activo de enrutamiento respetando cuantización mixta."""
        active_ids = [0, 15, 32, 58]
        routing_set = qwen_registry.get_routing_set(layer_id=3, expert_ids=active_ids)
        assert len(routing_set) == 4

        total_bytes = qwen_registry.get_routing_set_bytes(layer_id=3, expert_ids=active_ids)
        sample_exp = qwen_registry.get_expert(layer_id=3, expert_id=0)
        
        # En capa 3 con Q4_K_M los expertos son de 5,226,496 bytes
        assert total_bytes == 4 * sample_exp.total_bytes
        assert total_bytes == 20905984
        assert qwen_registry.get_routing_set_mb(layer_id=3, expert_ids=active_ids) == pytest.approx(19.937, rel=0.02)

    def test_index_out_of_bounds_errors(self, qwen_registry):
        """Valida el rechazo estricto de índices inválidos."""
        with pytest.raises(IndexError):
            qwen_registry.get_expert(layer_id=99, expert_id=0)

        with pytest.raises(IndexError):
            qwen_registry.get_expert(layer_id=0, expert_id=999)

    def test_memory_summary_export(self, qwen_registry):
        """Verifica que el resumen estructurado contenga todas las claves requeridas."""
        summary = qwen_registry.get_memory_summary()
        required_keys = [
            "model_name", "architecture", "total_layers", "experts_per_layer",
            "active_experts_per_token", "total_experts_count", "single_expert_bytes",
            "single_expert_mb", "routing_set_1layer_mb", "routing_set_all_layers_mb",
            "dense_memory_mb", "moe_memory_mb", "total_model_gb"
        ]
        for k in required_keys:
            assert k in summary
        assert summary["total_layers"] == 24
        assert summary["experts_per_layer"] == 60
