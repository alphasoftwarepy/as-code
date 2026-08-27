"""
AS-Core MoE Engine — VRAM Expert Pool
=====================================
Gestiona un pool pre-asignado de slots de memoria VRAM en la GPU para alojamiento,
subida, desalojo y reutilización O(1) de tensores de expertos individuales.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from core.moe.cuda_driver import CUDADriver
from core.moe.expert_tensor import ExpertTensor, ResidencyTier

logger = logging.getLogger("as-code.core.moe.vram_pool")


@dataclass
class VRAMSlot:
    """Representa un slot físico continuo de memoria en la VRAM de la GPU."""
    slot_id: int
    device_ptr: int
    capacity_bytes: int
    is_occupied: bool = False
    current_layer_id: Optional[int] = None
    current_expert_id: Optional[int] = None
    uploaded_bytes: int = 0
    upload_count: int = 0
    last_uploaded_at: float = 0.0

    @property
    def capacity_mb(self) -> float:
        return round(self.capacity_bytes / (1024 * 1024), 3)

    @property
    def is_free(self) -> bool:
        return not self.is_occupied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "device_ptr": hex(self.device_ptr),
            "capacity_bytes": self.capacity_bytes,
            "capacity_mb": self.capacity_mb,
            "is_occupied": self.is_occupied,
            "current_layer_id": self.current_layer_id,
            "current_expert_id": self.current_expert_id,
            "uploaded_bytes": self.uploaded_bytes,
            "upload_count": self.upload_count,
        }


class VRAMExpertPool:
    """Pool pre-asignado de slots en VRAM para tensores de expertos."""

    def __init__(
        self,
        num_slots: int,
        slot_size_bytes: int,
        cuda_driver: Optional[CUDADriver] = None,
        align_to_64b: bool = True,
    ):
        self.num_slots = num_slots
        # Alinear tamaño de slot a 64 bytes para accesos alineados en GPU
        if align_to_64b and (slot_size_bytes % 64 != 0):
            slot_size_bytes = ((slot_size_bytes + 63) // 64) * 64
        self.slot_size_bytes = slot_size_bytes

        self.driver = cuda_driver or CUDADriver()
        self._slots: List[VRAMSlot] = []
        self._expert_to_slot: Dict[Tuple[int, int], int] = {}

        # Métricas empíricas de operación
        self.stats = {
            "total_allocations": 0,
            "total_uploads": 0,
            "total_evictions": 0,
            "total_reuses": 0,
            "total_upload_time_ms": 0.0,
            "total_bytes_transferred": 0,
        }

        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Asigna los buffers contiguos de VRAM en la GPU para cada slot."""
        if not self.driver.is_initialized:
            logger.warning("[VRAMExpertPool] Driver CUDA no disponible. Operando en modo Mock/Simulado.")
            for i in range(self.num_slots):
                self._slots.append(
                    VRAMSlot(
                        slot_id=i,
                        device_ptr=0x10000000 + (i * self.slot_size_bytes),
                        capacity_bytes=self.slot_size_bytes,
                    )
                )
            return

        logger.info(
            f"[VRAMExpertPool] Inicializando pool de {self.num_slots} slots "
            f"({self.slot_size_bytes / (1024*1024):.2f} MB/slot, Total: {(self.num_slots * self.slot_size_bytes) / (1024*1024):.2f} MB VRAM)..."
        )

        for i in range(self.num_slots):
            d_ptr = self.driver.mem_alloc(self.slot_size_bytes)
            self._slots.append(
                VRAMSlot(
                    slot_id=i,
                    device_ptr=d_ptr,
                    capacity_bytes=self.slot_size_bytes,
                )
            )
            self.stats["total_allocations"] += 1

        logger.info(f"[VRAMExpertPool] Pool inicializado con éxito ({self.num_slots} slots activos).")

    @property
    def total_capacity_bytes(self) -> int:
        return self.num_slots * self.slot_size_bytes

    @property
    def total_capacity_mb(self) -> float:
        return round(self.total_capacity_bytes / (1024 * 1024), 2)

    @property
    def occupied_slots_count(self) -> int:
        return sum(1 for s in self._slots if s.is_occupied)

    @property
    def free_slots_count(self) -> int:
        return sum(1 for s in self._slots if s.is_free)

    def allocate_slot(self, layer_id: int, expert_id: int) -> int:
        """Reserva el primer slot libre disponible para el experto especificado."""
        key = (layer_id, expert_id)
        if key in self._expert_to_slot:
            return self._expert_to_slot[key]

        for slot in self._slots:
            if slot.is_free:
                slot.is_occupied = True
                slot.current_layer_id = layer_id
                slot.current_expert_id = expert_id
                self._expert_to_slot[key] = slot.slot_id
                return slot.slot_id

        raise MemoryError(f"[VRAMExpertPool] Pool agotado: {self.num_slots} slots ocupados. Se requiere desalojo previo.")

    def upload_expert(
        self,
        slot_id: int,
        expert: ExpertTensor,
        raw_bytes: Union[bytes, bytearray, memoryview, int],
    ) -> VRAMSlot:
        """Copia físicamente los pesos binarios del experto al slot de VRAM en la GPU."""
        if slot_id < 0 or slot_id >= len(self._slots):
            raise IndexError(f"Slot ID {slot_id} fuera de rango (0..{len(self._slots)-1})")

        slot = self._slots[slot_id]
        byte_len = expert.total_bytes if isinstance(raw_bytes, int) else len(raw_bytes)

        if byte_len > slot.capacity_bytes:
            raise ValueError(
                f"El experto {expert.layer_id}:{expert.expert_id} ({byte_len} bytes) "
                f"excede la capacidad del slot ({slot.capacity_bytes} bytes)"
            )

        # 1. Medir tiempo de transferencia física
        t0 = time.perf_counter()
        if self.driver.is_initialized:
            self.driver.memcpy_htod(slot.device_ptr, raw_bytes, byte_len)
            self.driver.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Actualizar estado del slot y del ExpertTensor
        slot.is_occupied = True
        slot.current_layer_id = expert.layer_id
        slot.current_expert_id = expert.expert_id
        slot.uploaded_bytes = byte_len
        slot.upload_count += 1
        slot.last_uploaded_at = time.time()

        expert.residency_tier = ResidencyTier.HOT
        expert.vram_device_ptr = slot.device_ptr
        self._expert_to_slot[(expert.layer_id, expert.expert_id)] = slot_id

        # 3. Métricas
        self.stats["total_uploads"] += 1
        self.stats["total_upload_time_ms"] += t_ms
        self.stats["total_bytes_transferred"] += byte_len

        return slot

    def evict_slot(self, slot_id: int) -> Optional[Tuple[int, int]]:
        """Marca un slot como disponible y desasocia al experto que residía en él."""
        if slot_id < 0 or slot_id >= len(self._slots):
            raise IndexError(f"Slot ID {slot_id} fuera de rango")

        slot = self._slots[slot_id]
        if not slot.is_occupied:
            return None

        evicted = (slot.current_layer_id, slot.current_expert_id)
        if evicted in self._expert_to_slot:
            del self._expert_to_slot[evicted]

        slot.is_occupied = False
        slot.current_layer_id = None
        slot.current_expert_id = None
        self.stats["total_evictions"] += 1

        return evicted

    def evict_expert(self, layer_id: int, expert_id: int, expert: Optional[ExpertTensor] = None) -> Optional[int]:
        """Desaloja a un experto específico por su ID de capa y experto."""
        key = (layer_id, expert_id)
        if key not in self._expert_to_slot:
            return None

        slot_id = self._expert_to_slot[key]
        self.evict_slot(slot_id)

        if expert is not None:
            expert.residency_tier = ResidencyTier.WARM if expert.ram_host_ptr else ResidencyTier.COLD
            expert.vram_device_ptr = None

        return slot_id

    def reuse_slot(
        self,
        slot_id: int,
        new_expert: ExpertTensor,
        new_raw_bytes: Union[bytes, bytearray, memoryview, int],
        old_expert: Optional[ExpertTensor] = None,
    ) -> VRAMSlot:
        """Reemplaza directamente el contenido de un slot existente con un nuevo experto
        sin pasar por desasignación ni reasignación de punteros en la GPU (Promoción en Caliente O(1)).
        """
        # Desalojar el experto anterior del mapa
        if old_expert is not None:
            old_expert.residency_tier = ResidencyTier.WARM if old_expert.ram_host_ptr else ResidencyTier.COLD
            old_expert.vram_device_ptr = None

        self.evict_slot(slot_id)
        slot = self.upload_expert(slot_id, new_expert, new_raw_bytes)
        self.stats["total_reuses"] += 1
        return slot

    def get_slot(self, slot_id: int) -> VRAMSlot:
        if slot_id < 0 or slot_id >= len(self._slots):
            raise IndexError(f"Slot ID {slot_id} fuera de rango")
        return self._slots[slot_id]

    def get_expert_slot(self, layer_id: int, expert_id: int) -> Optional[VRAMSlot]:
        key = (layer_id, expert_id)
        slot_id = self._expert_to_slot.get(key)
        return self._slots[slot_id] if slot_id is not None else None

    def find_slot(self, layer_id: int, expert_id: int) -> Optional[int]:
        """Devuelve el slot_id si el experto reside en VRAM, o None si es MISS."""
        return self._expert_to_slot.get((layer_id, expert_id))

    def find_free_slot(self) -> Optional[int]:
        """Devuelve el ID del primer slot libre, o None si todos están ocupados."""
        for slot in self._slots:
            if slot.is_free:
                return slot.slot_id
        return None

    def is_expert_resident(self, layer_id: int, expert_id: int) -> bool:
        """Determina si un experto está actualmente en VRAM."""
        return (layer_id, expert_id) in self._expert_to_slot

    @property
    def occupied_slots(self) -> int:
        return self.occupied_slots_count

    @property
    def allocated_bytes(self) -> int:
        return self.occupied_slots_count * self.slot_size_bytes

    @property
    def allocated_mb(self) -> float:
        return round(self.allocated_bytes / (1024 * 1024), 2)

    def get_pool_status(self) -> Dict[str, Any]:
        return {
            "num_slots": self.num_slots,
            "slot_size_bytes": self.slot_size_bytes,
            "slot_size_mb": round(self.slot_size_bytes / (1024 * 1024), 3),
            "total_capacity_mb": self.total_capacity_mb,
            "occupied_slots": self.occupied_slots_count,
            "free_slots": self.free_slots_count,
            "occupancy_pct": round((self.occupied_slots_count / self.num_slots) * 100, 1),
            "stats": self.stats,
        }

    def release(self) -> None:
        """Libera todos los buffers de VRAM al cerrar el motor."""
        if self.driver.is_initialized:
            for slot in self._slots:
                if slot.device_ptr != 0:
                    try:
                        self.driver.mem_free(slot.device_ptr)
                    except Exception as e:
                        logger.warning(f"Error liberando slot {slot.slot_id}: {e}")
                    slot.device_ptr = 0
        self._slots.clear()
        self._expert_to_slot.clear()
        logger.info("[VRAMExpertPool] Memoria VRAM liberada al 100%.")

    def __del__(self):
        self.release()
