"""
AS-Core MoE Engine — Residency Manager (B4.3.3)
==============================================
Conecta el RealRouter y ExpertRegistry con el VRAMExpertPool:
- Realiza Residency Lookup O(1) determinando HIT vs MISS para cada experto seleccionado.
- En caso de HIT: resuelve directamente el device_ptr en VRAM sin transferencias (0 bytes).
- En caso de MISS: recupera los bytes del experto y realiza la promoción física reutilizando slots VRAM.
- Gestiona el reemplazo de slots y registra métricas detalladas de hit rate, latencias y memoria.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.expert_tensor import ExpertTensor, ResidencyTier
from core.moe.ram_warm_pool import RAMWarmPool
from core.moe.router import RoutingDecision
from core.moe.vram_pool import VRAMExpertPool, VRAMSlot

logger = logging.getLogger("as-code.core.moe.residency")


@dataclass
class ResidencyDecision:
    """Resultado del proceso de residencia física para un experto individual."""
    layer_id: int
    expert_id: int
    is_hit: bool
    slot_id: int
    device_ptr: int
    size_bytes: int
    promotion_latency_ms: float = 0.0
    evicted_expert: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "is_hit": self.is_hit,
            "slot_id": self.slot_id,
            "device_ptr": hex(self.device_ptr),
            "promotion_latency_ms": round(self.promotion_latency_ms, 4),
            "evicted_expert": self.evicted_expert,
        }


@dataclass
class ResidencyLayerDispatch:
    """Resultado consolidado de residencia física para los 4 expertos activos de una capa."""
    layer_id: int
    decision: RoutingDecision
    residency_decisions: List[ResidencyDecision]
    total_promotion_latency_ms: float
    total_lookup_latency_ms: float
    total_vram_allocated_bytes: int

    @property
    def hit_count(self) -> int:
        return sum(1 for rd in self.residency_decisions if rd.is_hit)

    @property
    def miss_count(self) -> int:
        return sum(1 for rd in self.residency_decisions if not rd.is_hit)

    @property
    def hit_rate(self) -> float:
        total = len(self.residency_decisions)
        return (self.hit_count / total) if total > 0 else 0.0

    @property
    def expert_ids(self) -> List[int]:
        return [rd.expert_id for rd in self.residency_decisions]

    @property
    def slots(self) -> List[int]:
        return [rd.slot_id for rd in self.residency_decisions]

    @property
    def device_ptrs(self) -> List[int]:
        return [rd.device_ptr for rd in self.residency_decisions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_ids": self.expert_ids,
            "slots": self.slots,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_rate, 4),
            "total_promotion_latency_ms": round(self.total_promotion_latency_ms, 4),
            "total_lookup_latency_ms": round(self.total_lookup_latency_ms, 4),
            "total_vram_mb": round(self.total_vram_allocated_bytes / (1024 * 1024), 2),
            "decisions": [d.to_dict() for d in self.residency_decisions],
        }


class ResidencyManager:
    """Administrador de residencia física entre Router, Registry, RAM Warm Pool y VRAM Expert Pool."""

    def __init__(
        self,
        registry: ExpertRegistry,
        vram_pool: VRAMExpertPool,
        ram_pool: Optional[RAMWarmPool] = None,
    ):
        self.registry = registry
        self.vram_pool = vram_pool
        self.ram_pool = ram_pool

        # Puntero para política básica de reemplazo round-robin entre slots
        self._next_evict_slot: int = 0

        # Métricas acumuladas globales
        self.cumulative_hits: int = 0
        self.cumulative_misses: int = 0
        self.cumulative_promotions: int = 0
        self.cumulative_evictions: int = 0
        self.cumulative_promotion_time_ms: float = 0.0

    def lookup(self, layer_id: int, expert_id: int) -> Tuple[bool, Optional[int]]:
        """Determina de forma O(1) si el experto reside actualmente en algún slot de VRAM."""
        slot_id = self.vram_pool.find_slot(layer_id, expert_id)
        if slot_id is not None:
            return True, slot_id
        return False, None

    def get_expert_raw_bytes(self, layer_id: int, expert_id: int) -> bytes:
        """Obtiene de forma O(1) los bytes crudos del experto desde memmap o RAMWarmPool."""
        if self.ram_pool is not None and self.ram_pool.is_expert_cached(layer_id, expert_id):
            return self.ram_pool.get_slot_data(layer_id, expert_id)

        # Extracción directa desde memmap
        raw_tensors = self.registry.get_layer_raw_tensors(layer_id)
        gate_b = raw_tensors["gate"].data[expert_id].tobytes()
        up_b = raw_tensors["up"].data[expert_id].tobytes()
        down_b = raw_tensors["down"].data[expert_id].tobytes()
        return gate_b + up_b + down_b

    def ensure_in_vram(self, layer_id: int, expert_id: int) -> ResidencyDecision:
        """Garantiza la residencia física de un experto en VRAM resolviendo HITs y MISSes."""
        is_hit, slot_id = self.lookup(layer_id, expert_id)

        if is_hit and slot_id is not None:
            # === HIT PATH: 0 bytes transferidos, resolución directa de puntero ===
            self.cumulative_hits += 1
            slot = self.vram_pool.get_slot(slot_id)
            return ResidencyDecision(
                layer_id=layer_id,
                expert_id=expert_id,
                is_hit=True,
                slot_id=slot_id,
                device_ptr=slot.device_ptr,
                size_bytes=slot.capacity_bytes,
                promotion_latency_ms=0.0,
                evicted_expert=None,
            )

        # === MISS PATH: Promoción física a VRAM ===
        self.cumulative_misses += 1
        self.cumulative_promotions += 1
        expert = self.registry.get_expert(layer_id, expert_id)
        raw_data = self.get_expert_raw_bytes(layer_id, expert_id)

        t_prom_0 = time.perf_counter()

        # 1. Buscar slot libre
        target_slot_id = self.vram_pool.find_free_slot()
        evicted = None

        if target_slot_id is not None:
            # Asignación y carga inicial en slot libre
            self.vram_pool.allocate_slot(layer_id, expert_id)
            slot = self.vram_pool.upload_expert(target_slot_id, expert, raw_data)
        else:
            # 2. Todos los slots ocupados: Reutilización de slot (Round-Robin simple para B4.3.3)
            target_slot_id = self._next_evict_slot
            self._next_evict_slot = (self._next_evict_slot + 1) % self.vram_pool.num_slots

            victim_slot = self.vram_pool.get_slot(target_slot_id)
            evicted = (victim_slot.current_layer_id, victim_slot.current_expert_id)
            self.cumulative_evictions += 1

            slot = self.vram_pool.reuse_slot(target_slot_id, expert, raw_data)

        t_prom_ms = (time.perf_counter() - t_prom_0) * 1000.0
        self.cumulative_promotion_time_ms += t_prom_ms

        return ResidencyDecision(
            layer_id=layer_id,
            expert_id=expert_id,
            is_hit=False,
            slot_id=target_slot_id,
            device_ptr=slot.device_ptr,
            size_bytes=slot.capacity_bytes,
            promotion_latency_ms=t_prom_ms,
            evicted_expert=evicted,
        )

    def dispatch_routing(self, decision: RoutingDecision) -> ResidencyLayerDispatch:
        """Procesa los 4 expertos activos decididos dinámicamente por el Router."""
        t_lookup_0 = time.perf_counter()
        layer_id = decision.layer_id
        residency_decisions: List[ResidencyDecision] = []

        total_promotion_ms = 0.0

        for eid in decision.top_k_ids:
            rd = self.ensure_in_vram(layer_id, eid)
            residency_decisions.append(rd)
            total_promotion_ms += rd.promotion_latency_ms

        t_lookup_ms = (time.perf_counter() - t_lookup_0) * 1000.0

        return ResidencyLayerDispatch(
            layer_id=layer_id,
            decision=decision,
            residency_decisions=residency_decisions,
            total_promotion_latency_ms=total_promotion_ms,
            total_lookup_latency_ms=t_lookup_ms,
            total_vram_allocated_bytes=self.vram_pool.allocated_bytes,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Devuelve un resumen de métricas acumuladas de residencia."""
        total_requests = self.cumulative_hits + self.cumulative_misses
        hit_rate = (self.cumulative_hits / total_requests) if total_requests > 0 else 0.0
        return {
            "total_requests": total_requests,
            "hit_count": self.cumulative_hits,
            "miss_count": self.cumulative_misses,
            "hit_rate": round(hit_rate, 4),
            "promotion_count": self.cumulative_promotions,
            "promotion_time_total_ms": round(self.cumulative_promotion_time_ms, 3),
            "eviction_count": self.cumulative_evictions,
            "vram_slots_total": self.vram_pool.num_slots,
            "vram_slots_occupied": self.vram_pool.occupied_slots,
            "vram_allocated_mb": self.vram_pool.allocated_mb,
        }
