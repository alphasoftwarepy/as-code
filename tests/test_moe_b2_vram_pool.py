"""
AS-Core MoE Engine — Unit & Physical Tests for Bloque B2 (VRAM Expert Pool)
===========================================================================
Valida la asignación, subida física, desalojo y reutilización O(1) en VRAM
sobre la GPU NVIDIA GeForce GTX 1650 Ti (4 GB VRAM).
"""

import time
from pathlib import Path
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.expert_tensor import ExpertTensor, ResidencyTier
from core.moe.vram_pool import VRAMExpertPool, VRAMSlot

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"


@pytest.fixture(scope="module")
def shared_registry():
    return ExpertRegistry(QWEN_PATH)


@pytest.fixture(scope="module")
def shared_cuda_driver():
    driver = CUDADriver()
    yield driver
    driver.destroy()


class TestVRAMExpertPool:
    """Pruebas del pool de slots de memoria VRAM."""

    def test_pool_initialization_and_capacity(self, shared_cuda_driver):
        """Verifica que el pool se inicializa con los slots y tamaños requeridos."""
        slot_size = 6307840 # Tamaño exacto de 1 experto Qwen Q6_K
        pool = VRAMExpertPool(num_slots=4, slot_size_bytes=slot_size, cuda_driver=shared_cuda_driver)

        assert pool.num_slots == 4
        assert pool.slot_size_bytes >= slot_size
        assert pool.free_slots_count == 4
        assert pool.occupied_slots_count == 0
        assert pool.total_capacity_mb > 24.0

        pool.release()

    def test_lifecycle_allocate_upload_evict_reuse(self, shared_cuda_driver, shared_registry):
        """Prueba física del ciclo:
        1. Expert A -> Slot 0
        2. Expert B -> Slot 1
        3. Expert A -> Evict Slot 0
        4. Expert C -> Reuse Slot 0
        """
        slot_size = 6307840
        pool = VRAMExpertPool(num_slots=2, slot_size_bytes=slot_size, cuda_driver=shared_cuda_driver)

        # Obtener 3 expertos reales de Qwen
        exp_a = shared_registry.get_expert(layer_id=0, expert_id=1)
        exp_b = shared_registry.get_expert(layer_id=0, expert_id=2)
        exp_c = shared_registry.get_expert(layer_id=0, expert_id=3)

        dummy_bytes_a = b"\xAA" * exp_a.total_bytes
        dummy_bytes_b = b"\xBB" * exp_b.total_bytes
        dummy_bytes_c = b"\xCC" * exp_c.total_bytes

        # 1. Subir Expert A -> Slot 0
        slot_0_id = pool.allocate_slot(exp_a.layer_id, exp_a.expert_id)
        assert slot_0_id == 0
        slot_0 = pool.upload_expert(slot_0_id, exp_a, dummy_bytes_a)

        assert slot_0.is_occupied is True
        assert exp_a.residency_tier == ResidencyTier.HOT
        assert exp_a.vram_device_ptr == slot_0.device_ptr
        assert pool.occupied_slots_count == 1
        assert pool.free_slots_count == 1

        # 2. Subir Expert B -> Slot 1
        slot_1_id = pool.allocate_slot(exp_b.layer_id, exp_b.expert_id)
        assert slot_1_id == 1
        slot_1 = pool.upload_expert(slot_1_id, exp_b, dummy_bytes_b)

        assert slot_1.is_occupied is True
        assert exp_b.residency_tier == ResidencyTier.HOT
        assert pool.occupied_slots_count == 2
        assert pool.free_slots_count == 0

        # 3. Desalojar Expert A de Slot 0
        evicted = pool.evict_slot(slot_0_id)
        assert evicted == (exp_a.layer_id, exp_a.expert_id)
        assert slot_0.is_occupied is False
        assert pool.free_slots_count == 1

        # 4. Reutilizar Slot 0 con Expert C
        slot_c = pool.reuse_slot(slot_0_id, exp_c, dummy_bytes_c, old_expert=exp_a)
        assert slot_c.slot_id == slot_0_id
        assert slot_c.current_expert_id == exp_c.expert_id
        assert exp_c.residency_tier == ResidencyTier.HOT
        assert exp_a.residency_tier != ResidencyTier.HOT
        assert pool.occupied_slots_count == 2

        # Liberar recursos
        pool.release()

    def test_pool_exhaustion_raises_memory_error(self, shared_cuda_driver):
        """Verifica que el pool rechaza asignaciones cuando no hay slots libres."""
        pool = VRAMExpertPool(num_slots=2, slot_size_bytes=1024 * 1024, cuda_driver=shared_cuda_driver)

        slot0 = pool.allocate_slot(layer_id=0, expert_id=10)
        slot1 = pool.allocate_slot(layer_id=0, expert_id=20)
        assert slot0 == 0 and slot1 == 1

        with pytest.raises(MemoryError):
            pool.allocate_slot(layer_id=0, expert_id=30)

        pool.release()

    def test_physical_transfer_and_reuse_timings(self, shared_cuda_driver, shared_registry):
        """Mide experimentalmente los tiempos de upload y reuse en la GPU real."""
        slot_size = 6307840 # 6.02 MB
        pool = VRAMExpertPool(num_slots=4, slot_size_bytes=slot_size, cuda_driver=shared_cuda_driver)

        exp = shared_registry.get_expert(0, 5)
        raw_data = b"\x77" * exp.total_bytes

        # Medir tiempo de asignación + upload
        t_alloc_0 = time.perf_counter()
        slot_id = pool.allocate_slot(exp.layer_id, exp.expert_id)
        t_alloc_ms = (time.perf_counter() - t_alloc_0) * 1000.0

        t_up_0 = time.perf_counter()
        slot = pool.upload_expert(slot_id, exp, raw_data)
        t_up_ms = (time.perf_counter() - t_up_0) * 1000.0

        # Medir tiempo de desalojo
        t_evict_0 = time.perf_counter()
        pool.evict_slot(slot_id)
        t_evict_ms = (time.perf_counter() - t_evict_0) * 1000.0

        # Medir tiempo de reuse (promoción directa)
        exp_next = shared_registry.get_expert(0, 6)
        t_reuse_0 = time.perf_counter()
        pool.reuse_slot(slot_id, exp_next, raw_data)
        t_reuse_ms = (time.perf_counter() - t_reuse_0) * 1000.0

        print(f"\n[VRAM Metrics]: Alloc={t_alloc_ms:.4f}ms | Upload={t_up_ms:.4f}ms | Evict={t_evict_ms:.4f}ms | Reuse={t_reuse_ms:.4f}ms")

        assert t_alloc_ms < 1.0 # O(1) en RAM
        assert t_evict_ms < 1.0 # O(1) en RAM
        assert t_up_ms < 8.0    # Transferencia de 6MB en PCIe
        assert t_reuse_ms < 8.0 # Promoción directa en slot existente

        pool.release()
