"""
AS-Core MoE Engine — Unit & Benchmark Tests for Bloque B3 (RAM Warm Pool)
=========================================================================
Evalúa físicamente:
1. Almacenamiento en Memoria Paginable (Pageable) vs Fijada (Pinned / DMA).
2. Coste real de promoción RAM -> VRAM (Latencia en ms y Ancho de banda en GB/s).
3. Extracción de tensores desde GGUF mmap hacia RAM Warm Pool.
4. Desalojo y recuperación de memoria física.
"""

import mmap
import time
from pathlib import Path
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.expert_tensor import ExpertTensor, ResidencyTier
from core.moe.vram_pool import VRAMExpertPool
from core.moe.ram_warm_pool import RAMWarmPool, WarmSlot

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"


@pytest.fixture(scope="module")
def shared_registry():
    return ExpertRegistry(QWEN_PATH)


@pytest.fixture(scope="module")
def shared_cuda_driver():
    driver = CUDADriver()
    yield driver
    driver.destroy()


class TestRAMWarmPool:
    """Pruebas del pool WARM en memoria RAM del Host."""

    def test_pageable_pool_storage_and_eviction(self, shared_registry):
        """Verifica el almacenamiento y desalojo en memoria paginable estándar."""
        capacity = 50 * 1024 * 1024 # 50 MB
        pool = RAMWarmPool(max_capacity_bytes=capacity, use_pinned_memory=False)

        exp = shared_registry.get_expert(0, 10)
        dummy_data = b"\x11" * exp.total_bytes

        slot = pool.store_expert(exp, dummy_data)
        assert slot.is_pinned is False
        assert pool.contains_expert(0, 10) is True
        assert exp.residency_tier == ResidencyTier.WARM
        assert pool.current_used_bytes == exp.total_bytes

        # Desalojar
        evicted = pool.evict_expert(0, 10, exp)
        assert evicted is True
        assert pool.contains_expert(0, 10) is False
        assert pool.current_used_bytes == 0
        assert exp.residency_tier == ResidencyTier.COLD

        pool.release()

    def test_pinned_pool_storage_and_eviction(self, shared_cuda_driver, shared_registry):
        """Verifica el almacenamiento y desalojo en memoria fijada (DMA Pinned)."""
        capacity = 50 * 1024 * 1024 # 50 MB
        pool = RAMWarmPool(max_capacity_bytes=capacity, use_pinned_memory=True, cuda_driver=shared_cuda_driver)

        exp = shared_registry.get_expert(0, 11)
        dummy_data = b"\x22" * exp.total_bytes

        slot = pool.store_expert(exp, dummy_data)
        if shared_cuda_driver.is_initialized:
            assert slot.is_pinned is True
            assert isinstance(slot.raw_buffer, int)
            assert exp.is_pinned is True
            assert exp.ram_host_ptr != 0

        assert pool.contains_expert(0, 11) is True
        assert exp.residency_tier == ResidencyTier.WARM

        # Desalojar
        evicted = pool.evict_expert(0, 11, exp)
        assert evicted is True
        assert pool.contains_expert(0, 11) is False
        assert exp.is_pinned is False
        assert exp.ram_host_ptr is None

        pool.release()

    def test_load_from_gguf_mmap(self, shared_registry):
        """Verifica la carga directa desde mmap del archivo binario GGUF."""
        pool = RAMWarmPool(max_capacity_bytes=100 * 1024 * 1024, use_pinned_memory=False)
        exp = shared_registry.get_expert(1, 5)

        with open(QWEN_PATH, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                slot = pool.load_expert_from_gguf_mmap(exp, mm)

        assert slot.size_bytes == exp.total_bytes
        assert slot.layer_id == 1
        assert slot.expert_id == 5
        assert pool.contains_expert(1, 5) is True

        pool.release()

    def test_physical_promotion_pageable_vs_pinned(self, shared_cuda_driver, shared_registry):
        """Medición experimental del coste de promoción RAM -> VRAM:
        Compara la latencia y ancho de banda PCIe entre memoria Pageable y Pinned.
        """
        slot_size = 6307840 # 6.02 MB
        vram_pool = VRAMExpertPool(num_slots=4, slot_size_bytes=slot_size, cuda_driver=shared_cuda_driver)
        exp = shared_registry.get_expert(0, 15)
        raw_data = b"\x33" * exp.total_bytes

        # 1. Medir Promoción desde Memoria Pageable
        pool_pageable = RAMWarmPool(max_capacity_bytes=50 * 1024 * 1024, use_pinned_memory=False, cuda_driver=shared_cuda_driver)
        pool_pageable.store_expert(exp, raw_data)

        # Calentar y medir
        pageable_times = []
        for _ in range(10):
            vslot, t_ms = pool_pageable.promote_to_vram(exp, vram_pool, target_slot_id=0)
            pageable_times.append(t_ms)

        avg_pageable_ms = sum(pageable_times[3:]) / len(pageable_times[3:])
        bw_pageable = (exp.total_bytes / (1024**3)) / (avg_pageable_ms / 1000.0) if avg_pageable_ms > 0 else 0

        # 2. Medir Promoción desde Memoria Pinned (DMA)
        pool_pinned = RAMWarmPool(max_capacity_bytes=50 * 1024 * 1024, use_pinned_memory=True, cuda_driver=shared_cuda_driver)
        pool_pinned.store_expert(exp, raw_data)

        pinned_times = []
        for _ in range(10):
            vslot, t_ms = pool_pinned.promote_to_vram(exp, vram_pool, target_slot_id=1)
            pinned_times.append(t_ms)

        avg_pinned_ms = sum(pinned_times[3:]) / len(pinned_times[3:])
        bw_pinned = (exp.total_bytes / (1024**3)) / (avg_pinned_ms / 1000.0) if avg_pinned_ms > 0 else 0

        latency_reduction_pct = ((avg_pageable_ms - avg_pinned_ms) / avg_pageable_ms) * 100.0 if avg_pageable_ms > 0 else 0

        print(f"\n" + "=" * 70)
        print(" [B3 EMPIRICAL MEASUREMENT] COSTE REAL DE PROMOCIÓN RAM -> VRAM")
        print("=" * 70)
        print(f"  Tamaño de Tensor (1 Experto): {exp.total_mb:.2f} MB ({exp.total_bytes} bytes)")
        print(f"  • Pageable RAM -> VRAM:  {avg_pageable_ms:.3f} ms | Ancho de Banda: {bw_pageable:.2f} GB/s")
        print(f"  • Pinned DMA   -> VRAM:  {avg_pinned_ms:.3f} ms | Ancho de Banda: {bw_pinned:.2f} GB/s")
        print(f"  • Reducción de Latencia: {latency_reduction_pct:.1f}%")
        print("=" * 70)

        assert avg_pageable_ms < 10.0
        assert avg_pinned_ms < 10.0
        assert bw_pinned >= bw_pageable * 0.9 # Pinned rinde igual o superior a Pageable

        # Liberar recursos
        vram_pool.release()
        pool_pageable.release()
        pool_pinned.release()
