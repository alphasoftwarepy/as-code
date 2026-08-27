"""
AS-Core MoE Engine — Unit Tests for Dynamic LRU Residency Engine (B4.4.1 - B4.4.3)
==================================================================================
Valida:
1. Política LRU dinámica por capa con reemplazo O(1) in-place.
2. Distinción exacta de orígenes: VRAM_HIT, RAM_WARM_PINNED, RAM_WARM_PAGEABLE, NVME_COLD.
3. Presupuesto de memoria y escalabilidad: 4, 8 y 12 slots/capa en VRAM física.
4. Protección estricta contra OOM en la GPU GTX 1650 Ti (4 GB).
5. Mantenimiento y precisión de métricas en tiempo real.
"""

import time
import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.dynamic_residency_engine import (
    DynamicLayerDispatch,
    DynamicResidencyDecision,
    DynamicResidencyEngine,
    ResidencySource,
)
from core.moe.expert_registry import ExpertRegistry
from core.moe.ram_warm_pool import RAMWarmPool
from core.moe.router import RoutingDecision

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"


@pytest.fixture(scope="module")
def shared_registry():
    return ExpertRegistry(QWEN_PATH)


@pytest.fixture(scope="module")
def shared_driver():
    driver = CUDADriver()
    yield driver
    driver.destroy()


class TestDynamicResidencyEngine:
    """Pruebas del motor de residencia dinámica LRU (Fases B4.4.1 - B4.4.3)."""

    def test_b441_lru_eviction_and_reuse_per_layer(self, shared_registry, shared_driver):
        """Valida que la política LRU desaloje exactamente el experto menos recientemente usado."""
        # Motor con 4 slots por capa
        engine = DynamicResidencyEngine(
            registry=shared_registry,
            slots_per_layer=4,
            cuda_driver=shared_driver,
        )

        layer_id = 0

        # 1. Cargar expertos 1, 2, 3, 4 (Cold Misses -> 4 slots ocupados)
        for exp_id in [1, 2, 3, 4]:
            dec = engine.resolve_expert(layer_id, exp_id)
            assert dec.is_hit is False
            assert dec.source == ResidencySource.NVME_COLD
            assert dec.bytes_transferred > 0

        # 2. Acceder al experto 1 (debe ser HIT y convertirse en el Most Recently Used)
        dec_1_hit = engine.resolve_expert(layer_id, 1)
        assert dec_1_hit.is_hit is True
        assert dec_1_hit.source == ResidencySource.VRAM_HIT
        assert dec_1_hit.bytes_transferred == 0

        # En este punto el orden LRU (del más antiguo al más reciente) es: [2, 3, 4, 1]

        # 3. Solicitar un nuevo experto 5 (todos los 4 slots están llenos)
        # La víctima DEBE ser el experto 2 (el menos recientemente usado)
        dec_5 = engine.resolve_expert(layer_id, 5)
        assert dec_5.is_hit is False
        assert dec_5.evicted_expert == (0, 2)  # Capa 0, Experto 2 desalojado!

        # 4. Verificar que el experto 1 sigue en VRAM (HIT)
        dec_1_check = engine.resolve_expert(layer_id, 1)
        assert dec_1_check.is_hit is True

        # 5. Verificar que el experto 2 fue efectivamente desalojado (MISS)
        dec_2_check = engine.resolve_expert(layer_id, 2)
        assert dec_2_check.is_hit is False

        engine.release()

    def test_b442_warm_pinned_vs_pageable_vs_cold_path(self, shared_registry, shared_driver):
        """Valida la resolución jerárquica de orígenes (NVMe -> Pinned RAM -> Pageable RAM -> VRAM)."""
        sample_exp = shared_registry.get_expert(0, 0)

        # Crear RAM Warm Pool con capacidad para 4 expertos en Pinned RAM
        ram_pool = RAMWarmPool(
            max_capacity_bytes=sample_exp.total_bytes * 4,
            use_pinned_memory=True,
            cuda_driver=shared_driver,
        )

        # Cargar experto 10 en RAM Pinned
        raw_tensors = shared_registry.get_layer_raw_tensors(0)
        exp_10_bytes = (
            raw_tensors["gate"].data[10].tobytes()
            + raw_tensors["up"].data[10].tobytes()
            + raw_tensors["down"].data[10].tobytes()
        )
        exp_10 = shared_registry.get_expert(0, 10)
        ram_pool.store_expert(exp_10, exp_10_bytes)

        engine = DynamicResidencyEngine(
            registry=shared_registry,
            slots_per_layer=4,
            ram_pool=ram_pool,
            cuda_driver=shared_driver,
        )

        # Caso A: Experto 10 está en Pinned RAM -> RAM_WARM_PINNED
        dec_10 = engine.resolve_expert(0, 10)
        assert dec_10.is_hit is False
        assert dec_10.source == ResidencySource.RAM_WARM_PINNED
        assert dec_10.promotion_latency_ms > 0

        # Caso B: Siguiente acceso a Experto 10 -> VRAM_HIT
        dec_10_warm = engine.resolve_expert(0, 10)
        assert dec_10_warm.is_hit is True
        assert dec_10_warm.source == ResidencySource.VRAM_HIT
        assert dec_10_warm.promotion_latency_ms == 0.0

        # Caso C: Experto 50 no está en RAM -> NVME_COLD
        dec_50 = engine.resolve_expert(0, 50)
        assert dec_50.is_hit is False
        assert dec_50.source == ResidencySource.NVME_COLD

        engine.release()
        ram_pool.release()

    def test_b443_capacity_and_vram_budget_boundaries(self, shared_registry, shared_driver):
        """Valida que los tamaños 4, 8 y 12 slots/capa se inicialicen correctamente y >16 lance excepción."""
        # 4 slots/capa = 577.5 MB VRAM (PASS)
        e4 = DynamicResidencyEngine(registry=shared_registry, slots_per_layer=4, cuda_driver=shared_driver)
        assert e4.total_vram_mb == pytest.approx(577.5, abs=0.1)
        e4.release()

        # 8 slots/capa = 1155.1 MB VRAM (PASS)
        e8 = DynamicResidencyEngine(registry=shared_registry, slots_per_layer=8, cuda_driver=shared_driver)
        assert e8.total_vram_mb == pytest.approx(1155.1, abs=0.1)
        e8.release()

        # 12 slots/capa = 1732.65 MB VRAM (PASS)
        e12 = DynamicResidencyEngine(registry=shared_registry, slots_per_layer=12, cuda_driver=shared_driver)
        assert e12.total_vram_mb == pytest.approx(1732.6, abs=0.1)
        e12.release()

        # 16 slots/capa = 2310.2 MB VRAM -> Supera presupuesto de 3950 MB total (base 2211 + 2310 = 4521 MB) -> ValueError
        with pytest.raises(ValueError, match="excede el presupuesto de seguridad"):
            DynamicResidencyEngine(registry=shared_registry, slots_per_layer=16, cuda_driver=shared_driver)

    def test_b445_and_b446_profile_export_and_preloading(self, shared_registry, shared_driver, tmp_path):
        """Valida la exportación del perfil de residencia (B4.4.5) y la precarga en VRAM (B4.4.6)."""
        engine = DynamicResidencyEngine(
            registry=shared_registry,
            slots_per_layer=4,
            cuda_driver=shared_driver,
        )

        # 1. Simular actividad: resolver expertos en capas 0 y 1
        for _ in range(5):
            engine.resolve_expert(0, 10)
        for _ in range(3):
            engine.resolve_expert(0, 20)
        for _ in range(2):
            engine.resolve_expert(1, 5)

        # 2. Exportar perfil a archivo temporal
        profile_file = tmp_path / "test_profile.json"
        profile_data = engine.export_profile(str(profile_file))

        assert profile_file.exists()
        assert "hot_experts_per_layer" in profile_data
        assert profile_data["hot_experts_per_layer"]["0"][0] == 10  # Experto 10 fue el más frecuente
        assert profile_data["hot_experts_per_layer"]["0"][1] == 20

        engine.release()

        # 3. Precargar en un nuevo motor desde el perfil (B4.4.6)
        new_engine = DynamicResidencyEngine(
            registry=shared_registry,
            slots_per_layer=4,
            cuda_driver=shared_driver,
        )

        preloaded_count = new_engine.preload_from_profile(str(profile_file), top_n_per_layer=2)
        assert preloaded_count >= 3  # Expertos 10, 20 (capa 0) y 5 (capa 1) precargados

        # 4. Verificar que el acceso a los expertos precargados es un VRAM HIT inmediato
        dec_10 = new_engine.resolve_expert(0, 10)
        assert dec_10.is_hit is True
        assert dec_10.source == ResidencySource.VRAM_HIT
        assert dec_10.bytes_transferred == 0

        dec_5 = new_engine.resolve_expert(1, 5)
        assert dec_5.is_hit is True
        assert dec_5.source == ResidencySource.VRAM_HIT

        new_engine.release()
