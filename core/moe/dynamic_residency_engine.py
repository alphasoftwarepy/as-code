"""
AS-Core MoE Engine — Dynamic LRU Residency Manager & Engine (B4.4)
===================================================================
Motor de residencia física dinámica de alto rendimiento para modelos MoE:
- Política LRU dinámica por capa agnóstica al modelo (Qwen, OLMoE, DeepSeek, etc.).
- Jerarquía de 3 niveles:
    1. VRAM HIT (0 bytes, 0 latency).
    2. RAM WARM MISS (Pinned DMA transfer a 10.4 GB/s o Pageable fallback).
    3. NVMe COLD MISS (Extracción directa desde GGUF memmap).
- Reutilización de slots O(1) in-place sin cuMemFree/cuMemAlloc por token.
- Rastreo de frecuencia y recencia en tiempo real.
- Telemetría exhaustiva de memoria, latencias y ancho de banda PCIe.
"""

from __future__ import annotations

import ctypes
import enum
import logging
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.expert_tensor import ExpertTensor, ResidencyTier
from core.moe.model_profile import ModelProfile
from core.moe.ram_warm_pool import RAMWarmPool
from core.moe.router import RoutingDecision
from core.moe.vram_pool import VRAMExpertPool, VRAMSlot

logger = logging.getLogger("as-code.core.moe.dynamic_residency")


class ResidencySource(str, enum.Enum):
    """Origen de resolución del tensor del experto."""
    VRAM_HIT = "VRAM_HIT"
    RAM_WARM_PINNED = "RAM_WARM_PINNED"
    RAM_WARM_PAGEABLE = "RAM_WARM_PAGEABLE"
    NVME_COLD = "NVME_COLD"


@dataclass
class DynamicResidencyDecision:
    """Decisión y métricas individuales para la resolución de un experto en VRAM."""
    layer_id: int
    expert_id: int
    is_hit: bool
    source: ResidencySource
    slot_id: int
    device_ptr: int
    size_bytes: int
    bytes_transferred: int
    promotion_latency_ms: float = 0.0
    eviction_latency_ms: float = 0.0
    evicted_expert: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "is_hit": self.is_hit,
            "source": self.source.value,
            "slot_id": self.slot_id,
            "device_ptr": hex(self.device_ptr),
            "bytes_transferred": self.bytes_transferred,
            "promotion_latency_ms": round(self.promotion_latency_ms, 4),
            "eviction_latency_ms": round(self.eviction_latency_ms, 4),
            "evicted_expert": self.evicted_expert,
        }


@dataclass
class DynamicLayerDispatch:
    """Resultado de resolución de residencia para los K expertos de una capa."""
    layer_id: int
    decision: RoutingDecision
    residency_decisions: List[DynamicResidencyDecision]
    total_promotion_latency_ms: float
    total_lookup_latency_ms: float
    total_bytes_transferred: int

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
            "bytes_transferred": self.total_bytes_transferred,
            "promotion_latency_ms": round(self.total_promotion_latency_ms, 4),
            "lookup_latency_ms": round(self.total_lookup_latency_ms, 4),
            "decisions": [d.to_dict() for d in self.residency_decisions],
        }


class DynamicResidencyEngine:
    """Motor central de residencia dinámica LRU por capa con jerarquía VRAM/RAM/NVMe."""

    def __init__(
        self,
        registry: ExpertRegistry,
        slots_per_layer: int = 8,
        ram_pool: Optional[RAMWarmPool] = None,
        cuda_driver: Optional[CUDADriver] = None,
        max_vram_budget_mb: float = 3950.0,
    ):
        self.registry = registry
        self.profile = self.registry.profile
        self.num_layers = self.profile.block_count
        self.num_experts_per_layer = self.profile.expert_count
        self.k_active = self.profile.expert_used_count
        self.driver = cuda_driver or CUDADriver()
        self.ram_pool = ram_pool

        self.slots_per_layer = slots_per_layer
        self.total_slots = self.num_layers * self.slots_per_layer
        self.slot_size_bytes = self.registry.get_expert(0, 0).total_bytes
        self.max_vram_budget_mb = max_vram_budget_mb

        self.vram_pool: Optional[VRAMExpertPool] = None
        self._validate_vram_budget()

        # Estructuras de LRU por capa: [layer_id -> OrderedDict(expert_id -> slot_id)]
        self._layer_lru: List[OrderedDict[int, int]] = [OrderedDict() for _ in range(self.num_layers)]

        # Slots libres por capa: [layer_id -> List[slot_id]]
        self._layer_free_slots: List[List[int]] = [
            list(range(l * self.slots_per_layer, (l + 1) * self.slots_per_layer))
            for l in range(self.num_layers)
        ]

        # VRAM Pool global segmentado
        self.vram_pool = VRAMExpertPool(
            num_slots=self.total_slots,
            slot_size_bytes=self.slot_size_bytes,
            cuda_driver=self.driver,
        )

        # Estadísticas y Frecuencia en tiempo real
        self.expert_frequency: Counter[Tuple[int, int]] = Counter()
        self.expert_last_access_ts: Dict[Tuple[int, int], int] = {}

        # Contadores de telemetría globales
        self.stats = {
            "total_requests": 0,
            "vram_hits": 0,
            "ram_pinned_hits": 0,
            "ram_pageable_hits": 0,
            "nvme_cold_misses": 0,
            "evictions": 0,
            "promotions": 0,
            "bytes_h2d": 0,
            "bytes_d2h": 0,
            "total_promotion_latency_ms": 0.0,
            "total_eviction_latency_ms": 0.0,
            "total_lookup_latency_ms": 0.0,
        }

        logger.info(
            f"[DynamicResidencyEngine] Inicializado con {self.slots_per_layer} slots/capa "
            f"({self.total_slots} slots totales = {self.total_vram_mb:.2f} MB VRAM)."
        )

    def _validate_vram_budget(self) -> None:
        """Verifica que la asignación no sobrepase el presupuesto físico de 4 GB."""
        moe_vram_mb = (self.total_slots * self.slot_size_bytes) / (1024 * 1024)
        base_vram_mb = 2211.25 # Attention + KV 2K + 24 Routers + OS/CUDA overhead
        total_projected_mb = base_vram_mb + moe_vram_mb

        if total_projected_mb > self.max_vram_budget_mb:
            raise ValueError(
                f"La capacidad de {self.slots_per_layer} slots/capa ({total_projected_mb:.1f} MB VRAM) "
                f"excede el presupuesto de seguridad de {self.max_vram_budget_mb:.1f} MB en la GTX 1650 Ti."
            )

    @property
    def total_vram_mb(self) -> float:
        return round((self.total_slots * self.slot_size_bytes) / (1024 * 1024), 2)

    def lookup(self, layer_id: int, expert_id: int) -> Tuple[bool, Optional[int]]:
        """Comprueba de forma O(1) si el experto está residente en el LRU de la capa."""
        lru = self._layer_lru[layer_id]
        if expert_id in lru:
            return True, lru[expert_id]
        return False, None

    def get_expert_source_and_bytes(self, layer_id: int, expert_id: int) -> Tuple[ResidencySource, bytes]:
        """Recupera los bytes crudos del experto y clasifica el origen (Pinned, Pageable, NVMe)."""
        if self.ram_pool is not None and self.ram_pool.contains_expert(layer_id, expert_id):
            slot = self.ram_pool.get_slot(layer_id, expert_id)
            if slot is not None:
                data = slot.raw_buffer
                if isinstance(data, int):
                    # Es un puntero pinned host, extraer bytes
                    buf = (ctypes.c_char * slot.size_bytes).from_address(data)
                    raw_bytes = bytes(buf)
                else:
                    raw_bytes = bytes(data)
                source = ResidencySource.RAM_WARM_PINNED if slot.is_pinned else ResidencySource.RAM_WARM_PAGEABLE
                return source, raw_bytes

        # NVMe / GGUF Memmap fallback
        raw_tensors = self.registry.get_layer_raw_tensors(layer_id)
        gate_b = raw_tensors["gate"].data[expert_id].tobytes()
        up_b = raw_tensors["up"].data[expert_id].tobytes()
        down_b = raw_tensors["down"].data[expert_id].tobytes()
        return ResidencySource.NVME_COLD, (gate_b + up_b + down_b)

    def resolve_expert(self, layer_id: int, expert_id: int) -> DynamicResidencyDecision:
        """Resuelve la residencia física de un experto individual aplicando política LRU."""
        t_look_0 = time.perf_counter()
        self.stats["total_requests"] += 1
        self.expert_frequency[(layer_id, expert_id)] += 1
        self.expert_last_access_ts[(layer_id, expert_id)] = time.perf_counter_ns()

        lru = self._layer_lru[layer_id]

        # ── 1. VRAM HIT PATH (0 Bytes, 0 Latencia de transferencia) ──────────
        if expert_id in lru:
            self.stats["vram_hits"] += 1
            slot_id = lru[expert_id]
            # Mover al final de la cola LRU (Most Recently Used)
            lru.move_to_end(expert_id)
            slot = self.vram_pool.get_slot(slot_id)

            t_look_ms = (time.perf_counter() - t_look_0) * 1000.0
            self.stats["total_lookup_latency_ms"] += t_look_ms

            return DynamicResidencyDecision(
                layer_id=layer_id,
                expert_id=expert_id,
                is_hit=True,
                source=ResidencySource.VRAM_HIT,
                slot_id=slot_id,
                device_ptr=slot.device_ptr,
                size_bytes=slot.capacity_bytes,
                bytes_transferred=0,
                promotion_latency_ms=0.0,
                eviction_latency_ms=0.0,
                evicted_expert=None,
            )

        # ── 2. MISS PATH (Promoción física a VRAM vía LRU) ───────────────────
        source, raw_bytes = self.get_expert_source_and_bytes(layer_id, expert_id)
        if source == ResidencySource.RAM_WARM_PINNED:
            self.stats["ram_pinned_hits"] += 1
        elif source == ResidencySource.RAM_WARM_PAGEABLE:
            self.stats["ram_pageable_hits"] += 1
        else:
            self.stats["nvme_cold_misses"] += 1

        self.stats["promotions"] += 1
        expert = self.registry.get_expert(layer_id, expert_id)

        free_slots = self._layer_free_slots[layer_id]
        evicted = None
        t_evict_ms = 0.0

        t_prom_0 = time.perf_counter()

        if free_slots:
            # Slot libre disponible en esta capa
            target_slot_id = free_slots.pop(0)
            slot = self.vram_pool.upload_expert(target_slot_id, expert, raw_bytes)
        else:
            # Desalojar el experto Menos Recientemente Usado (LRU Head)
            t_ev_0 = time.perf_counter()
            victim_expert_id, victim_slot_id = lru.popitem(last=False)
            target_slot_id = victim_slot_id
            evicted = (layer_id, victim_expert_id)
            self.stats["evictions"] += 1
            t_evict_ms = (time.perf_counter() - t_ev_0) * 1000.0
            self.stats["total_eviction_latency_ms"] += t_evict_ms

            # Reutilización in-place en VRAM sin reasignar memoria
            slot = self.vram_pool.reuse_slot(target_slot_id, expert, raw_bytes)

        # Registrar en la cola LRU como Most Recently Used
        lru[expert_id] = target_slot_id

        t_prom_ms = (time.perf_counter() - t_prom_0) * 1000.0
        self.stats["total_promotion_latency_ms"] += t_prom_ms
        self.stats["bytes_h2d"] += len(raw_bytes)

        t_look_ms = (time.perf_counter() - t_look_0) * 1000.0
        self.stats["total_lookup_latency_ms"] += t_look_ms

        return DynamicResidencyDecision(
            layer_id=layer_id,
            expert_id=expert_id,
            is_hit=False,
            source=source,
            slot_id=target_slot_id,
            device_ptr=slot.device_ptr,
            size_bytes=slot.capacity_bytes,
            bytes_transferred=len(raw_bytes),
            promotion_latency_ms=t_prom_ms,
            eviction_latency_ms=t_evict_ms,
            evicted_expert=evicted,
        )

    def dispatch_layer(self, decision: RoutingDecision) -> DynamicLayerDispatch:
        """Despacha la resolución de los K expertos activos para una capa en un token."""
        t0 = time.perf_counter()
        layer_id = decision.layer_id
        decisions: List[DynamicResidencyDecision] = []
        total_prom_ms = 0.0
        total_bytes = 0

        for eid in decision.top_k_ids:
            rd = self.resolve_expert(layer_id, eid)
            decisions.append(rd)
            total_prom_ms += rd.promotion_latency_ms
            total_bytes += rd.bytes_transferred

        t_total_ms = (time.perf_counter() - t0) * 1000.0

        return DynamicLayerDispatch(
            layer_id=layer_id,
            decision=decision,
            residency_decisions=decisions,
            total_promotion_latency_ms=total_prom_ms,
            total_lookup_latency_ms=t_total_ms,
            total_bytes_transferred=total_bytes,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Calcula el informe consolidado de métricas de rendimiento y memoria."""
        total = self.stats["total_requests"]
        vram_hits = self.stats["vram_hits"]
        hit_rate = (vram_hits / total) if total > 0 else 0.0
        miss_rate = 1.0 - hit_rate

        return {
            "total_requests": total,
            "vram_hits": vram_hits,
            "hit_rate": round(hit_rate, 4),
            "miss_rate": round(miss_rate, 4),
            "ram_pinned_hits": self.stats["ram_pinned_hits"],
            "ram_pageable_hits": self.stats["ram_pageable_hits"],
            "nvme_cold_misses": self.stats["nvme_cold_misses"],
            "promotions": self.stats["promotions"],
            "evictions": self.stats["evictions"],
            "bytes_h2d_mb": round(self.stats["bytes_h2d"] / (1024 * 1024), 2),
            "avg_promotion_latency_ms": round(
                (self.stats["total_promotion_latency_ms"] / self.stats["promotions"])
                if self.stats["promotions"] > 0
                else 0.0,
                4,
            ),
            "slots_per_layer": self.slots_per_layer,
            "total_slots": self.total_slots,
            "vram_allocated_mb": self.total_vram_mb,
        }

    def export_profile(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Exporta el perfil de frecuencia y recencia acumulado por capa (B4.4.5)."""
        import datetime
        import json
        from pathlib import Path

        hot_experts_per_layer: Dict[str, List[int]] = {}
        layer_details: Dict[str, Dict[str, Any]] = {}

        for l in range(self.num_layers):
            exp_counts = [
                (eid, self.expert_frequency.get((l, eid), 0))
                for eid in range(self.num_experts_per_layer)
            ]
            exp_counts.sort(key=lambda x: x[1], reverse=True)
            hot_eids = [eid for eid, c in exp_counts if c > 0]
            hot_experts_per_layer[str(l)] = hot_eids[: self.slots_per_layer]

            total_accesses_layer = sum(c for _, c in exp_counts)
            layer_details[str(l)] = {
                "total_layer_requests": total_accesses_layer,
                "active_unique_experts": len(hot_eids),
                "top_experts": [
                    {"expert_id": eid, "frequency": c, "relative_pct": round(c / total_accesses_layer, 4) if total_accesses_layer > 0 else 0.0}
                    for eid, c in exp_counts[: self.slots_per_layer]
                ],
            }

        profile_data = {
            "model_path": self.profile.model_path,
            "architecture": self.profile.architecture,
            "block_count": self.num_layers,
            "expert_count": self.num_experts_per_layer,
            "expert_used_count": self.k_active,
            "slots_per_layer": self.slots_per_layer,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "global_metrics": self.get_metrics(),
            "hot_experts_per_layer": hot_experts_per_layer,
            "layer_stats": layer_details,
        }

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2)
            logger.info(f"[DynamicResidencyEngine] Perfil exportado a {output_path}")

        return profile_data

    def preload_from_profile(
        self,
        profile_source: Union[str, Dict[str, Any]],
        top_n_per_layer: Optional[int] = None,
    ) -> int:
        """Precarga en VRAM los Top-N expertos aprendidos por capa para mitigar el Cold Start (B4.4.6)."""
        import json
        from pathlib import Path

        if isinstance(profile_source, (str, Path)):
            with open(profile_source, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
        else:
            profile_data = profile_source

        hot_per_layer = profile_data.get("hot_experts_per_layer", {})
        limit = top_n_per_layer if top_n_per_layer is not None else self.slots_per_layer
        preloaded_count = 0

        logger.info(f"[DynamicResidencyEngine] Precargando hasta {limit} expertos por capa en VRAM...")

        for l in range(self.num_layers):
            layer_key = str(l)
            if layer_key in hot_per_layer:
                target_eids = hot_per_layer[layer_key][:limit]
                for eid in target_eids:
                    # Garantizar que el experto esté cargado en VRAM
                    if eid not in self._layer_lru[l]:
                        self.resolve_expert(l, eid)
                        preloaded_count += 1

        logger.info(f"[DynamicResidencyEngine] Precarga completada: {preloaded_count} expertos en VRAM.")
        return preloaded_count

    def reset_metrics(self) -> None:
        """Reinicia los contadores de telemetría sin desalojar los slots de VRAM."""
        for k in self.stats:
            if isinstance(self.stats[k], int):
                self.stats[k] = 0
            elif isinstance(self.stats[k], float):
                self.stats[k] = 0.0

    def release(self) -> None:
        """Libera todos los recursos de GPU."""
        if hasattr(self, "vram_pool") and self.vram_pool is not None:
            self.vram_pool.release()
            self.vram_pool = None
        if hasattr(self, "_layer_lru"):
            for lru in self._layer_lru:
                lru.clear()
        if hasattr(self, "_layer_free_slots"):
            self._layer_free_slots.clear()
        logger.info("[DynamicResidencyEngine] Todos los slots de VRAM han sido liberados.")

    def __del__(self):
        self.release()
