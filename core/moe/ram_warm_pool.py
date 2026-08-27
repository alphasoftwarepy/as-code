"""
AS-Core MoE Engine — RAM Warm Pool (Tier 2)
===========================================
Gestiona el almacenamiento de segundo nivel (WARM) en la memoria RAM del Host.
Implementa dos estrategias físicas de memoria:
1. Memoria Paginable (Pageable): Estándar de sistema, menor consumo de recursos bloqueados.
2. Memoria Fijada (Pinned / DMA): Memoria bloqueada físicamente con cuMemAllocHost para transferencias PCIe directas a máxima velocidad.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from core.moe.cuda_driver import CUDADriver
from core.moe.expert_tensor import ExpertTensor, ResidencyTier
from core.moe.vram_pool import VRAMExpertPool, VRAMSlot

logger = logging.getLogger("as-code.core.moe.warm_pool")


@dataclass
class WarmSlot:
    """Representa un slot de almacenamiento de un experto en la memoria RAM del Host."""
    slot_id: int
    layer_id: int
    expert_id: int
    size_bytes: int
    is_pinned: bool
    # Si es pinned, host_ptr es el puntero entero de 64-bits; si es pageable, es un bytearray
    raw_buffer: Union[bytearray, int]
    loaded_at: float = 0.0
    access_count: int = 0

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 3)

    @property
    def host_address(self) -> int:
        if self.is_pinned and isinstance(self.raw_buffer, int):
            return self.raw_buffer
        elif isinstance(self.raw_buffer, bytearray):
            return ctypes.addressof((ctypes.c_char * len(self.raw_buffer)).from_buffer(self.raw_buffer))
        return 0


class RAMWarmPool:
    """Pool de almacenamiento WARM en Host RAM para expertos no residentes en VRAM."""

    def __init__(
        self,
        max_capacity_bytes: int,
        use_pinned_memory: bool = False,
        cuda_driver: Optional[CUDADriver] = None,
    ):
        self.max_capacity_bytes = max_capacity_bytes
        self.use_pinned_memory = use_pinned_memory
        self.driver = cuda_driver or CUDADriver()

        # Diccionario indexado: (layer_id, expert_id) -> WarmSlot
        self._slots: Dict[Tuple[int, int], WarmSlot] = {}
        self._current_used_bytes: int = 0
        self._next_slot_id: int = 0

        # Métricas empíricas de operación
        self.stats = {
            "stored_experts_count": 0,
            "total_promotions_to_vram": 0,
            "total_evictions": 0,
            "total_promotion_time_ms": 0.0,
            "total_promoted_bytes": 0,
            "pinned_allocations": 0,
            "pageable_allocations": 0,
        }

        mode_str = "PINNED (DMA Acelerado)" if self.use_pinned_memory else "PAGEABLE (Estándar)"
        logger.info(
            f"[RAMWarmPool] Inicializado modo {mode_str}. Capacidad máxima: {self.max_capacity_bytes / (1024*1024):.2f} MB."
        )

    @property
    def current_used_bytes(self) -> int:
        return self._current_used_bytes

    @property
    def current_used_mb(self) -> float:
        return round(self._current_used_bytes / (1024 * 1024), 2)

    @property
    def total_capacity_mb(self) -> float:
        return round(self.max_capacity_bytes / (1024 * 1024), 2)

    @property
    def expert_count(self) -> int:
        return len(self._slots)

    def contains_expert(self, layer_id: int, expert_id: int) -> bool:
        return (layer_id, expert_id) in self._slots

    def store_expert(
        self,
        expert: ExpertTensor,
        raw_data: Union[bytes, bytearray, memoryview],
    ) -> WarmSlot:
        """Almacena los datos binarios de un experto en la memoria RAM del Host."""
        key = (expert.layer_id, expert.expert_id)
        if key in self._slots:
            return self._slots[key]

        data_len = len(raw_data)
        if self._current_used_bytes + data_len > self.max_capacity_bytes:
            raise MemoryError(
                f"[RAMWarmPool] Capacidad excedida ({self.current_used_mb} MB / {self.total_capacity_mb} MB). Se requiere desalojo."
            )

        slot_id = self._next_slot_id
        self._next_slot_id += 1

        if self.use_pinned_memory and self.driver.is_initialized:
            # Asignación en memoria fijada (DMA-locked)
            h_ptr = self.driver.mem_alloc_host(data_len)
            c_src = (ctypes.c_char * data_len).from_buffer_copy(raw_data)
            ctypes.memmove(h_ptr, c_src, data_len)
            raw_buf: Union[bytearray, int] = h_ptr
            is_pinned = True
            self.stats["pinned_allocations"] += 1
            expert.ram_host_ptr = h_ptr
            expert.is_pinned = True
        else:
            # Asignación en memoria paginable
            raw_buf = bytearray(raw_data)
            is_pinned = False
            self.stats["pageable_allocations"] += 1
            expert.ram_host_ptr = None
            expert.is_pinned = False

        slot = WarmSlot(
            slot_id=slot_id,
            layer_id=expert.layer_id,
            expert_id=expert.expert_id,
            size_bytes=data_len,
            is_pinned=is_pinned,
            raw_buffer=raw_buf,
            loaded_at=time.time(),
        )

        self._slots[key] = slot
        self._current_used_bytes += data_len
        self.stats["stored_experts_count"] += 1

        # Actualizar nivel de residencia si no está HOT en VRAM
        if expert.residency_tier != ResidencyTier.HOT:
            expert.residency_tier = ResidencyTier.WARM

        return slot

    def load_expert_from_gguf_mmap(
        self,
        expert: ExpertTensor,
        mmap_handle: mmap.mmap,
    ) -> WarmSlot:
        """Extrae de forma eficiente los slices del experto desde el mmap del GGUF a RAM."""
        slices = expert.get_slices()
        total_b = expert.total_bytes
        temp_buf = bytearray(total_b)
        offset = 0

        for s in slices:
            mmap_handle.seek(s.offset_in_gguf)
            slice_bytes = mmap_handle.read(s.n_bytes)
            temp_buf[offset : offset + s.n_bytes] = slice_bytes
            offset += s.n_bytes

        return self.store_expert(expert, temp_buf)

    def get_slot(self, layer_id: int, expert_id: int) -> Optional[WarmSlot]:
        key = (layer_id, expert_id)
        slot = self._slots.get(key)
        if slot:
            slot.access_count += 1
        return slot

    def promote_to_vram(
        self,
        expert: ExpertTensor,
        vram_pool: VRAMExpertPool,
        target_slot_id: Optional[int] = None,
    ) -> Tuple[VRAMSlot, float]:
        """Transfiere físicamente el experto desde el pool WARM (Host RAM)
        hacia un slot de la GPU VRAM (HOT), midiendo la latencia exacta de promoción.
        """
        key = (expert.layer_id, expert.expert_id)
        warm_slot = self._slots.get(key)
        if warm_slot is None:
            raise KeyError(f"El experto {expert.layer_id}:{expert.expert_id} no reside en RAM Warm Pool.")

        warm_slot.access_count += 1

        # Asignar o reutilizar slot en VRAM
        if target_slot_id is None:
            target_slot_id = vram_pool.allocate_slot(expert.layer_id, expert.expert_id)

        t_start = time.perf_counter()

        if warm_slot.is_pinned and isinstance(warm_slot.raw_buffer, int):
            # Transferencia acelerada por DMA directo
            vslot = vram_pool.upload_expert(target_slot_id, expert, warm_slot.raw_buffer)
        else:
            # Transferencia paginable estándar
            vslot = vram_pool.upload_expert(target_slot_id, expert, warm_slot.raw_buffer)

        promotion_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Actualizar métricas
        self.stats["total_promotions_to_vram"] += 1
        self.stats["total_promotion_time_ms"] += promotion_time_ms
        self.stats["total_promoted_bytes"] += warm_slot.size_bytes

        return vslot, promotion_time_ms

    def evict_expert(self, layer_id: int, expert_id: int, expert: Optional[ExpertTensor] = None) -> bool:
        """Desaloja un experto de la RAM del Host y libera sus buffers asociados."""
        key = (layer_id, expert_id)
        if key not in self._slots:
            return False

        slot = self._slots.pop(key)
        self._current_used_bytes -= slot.size_bytes

        if slot.is_pinned and isinstance(slot.raw_buffer, int) and self.driver.is_initialized:
            try:
                self.driver.mem_free_host(slot.raw_buffer)
            except Exception as e:
                logger.warning(f"Error liberando memoria fijada para slot {slot.slot_id}: {e}")

        if expert is not None:
            expert.ram_host_ptr = None
            expert.is_pinned = False
            if expert.residency_tier == ResidencyTier.WARM:
                expert.residency_tier = ResidencyTier.COLD

        self.stats["total_evictions"] += 1
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "PINNED (DMA)" if self.use_pinned_memory else "PAGEABLE",
            "expert_count": self.expert_count,
            "used_memory_mb": self.current_used_mb,
            "total_capacity_mb": self.total_capacity_mb,
            "occupancy_pct": round((self._current_used_bytes / max(1, self.max_capacity_bytes)) * 100, 1),
            "stats": self.stats,
        }

    def release(self) -> None:
        """Libera todos los buffers y memoria fijada de Host al cerrar."""
        if self.use_pinned_memory and self.driver.is_initialized:
            for slot in list(self._slots.values()):
                if slot.is_pinned and isinstance(slot.raw_buffer, int):
                    try:
                        self.driver.mem_free_host(slot.raw_buffer)
                    except Exception:
                        pass
        self._slots.clear()
        self._current_used_bytes = 0
        logger.info("[RAMWarmPool] Recursos de RAM liberados al 100%.")

    def __del__(self):
        self.release()
