"""
AS-Core MoE Engine — Expert Tensor Abstraction
===============================================
Representa la abstracción de bajo nivel de tensores y slices individuales de un experto MoE.
Diseño completamente agnóstico al modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ResidencyTier(str, Enum):
    """Nivel de residencia física de un tensor/experto."""
    COLD = "cold"   # Residente exclusivamente en disco NVMe / Storage
    WARM = "warm"   # Residente en RAM del Host (Pageable o Pinned)
    HOT = "hot"     # Residente en VRAM de la GPU


class TensorRole(str, Enum):
    """Rol funcional del tensor dentro de la arquitectura Transformer MoE."""
    GATE = "gate"               # Proyección de compuerta (ffn_gate_exps)
    UP = "up"                   # Proyección ascendente (ffn_up_exps)
    DOWN = "down"               # Proyección descendente (ffn_down_exps)
    SHARED = "shared"           # Experto compartido / FFN denso común
    ROUTER = "router"           # Tensor de pesos del Router / Gate classifier
    ATTENTION = "attention"     # Atenciones densas (Q, K, V, O)
    EMBEDDING = "embedding"     # Embeddings / Output Head
    NORM = "norm"               # Normalizaciones (LayerNorm, RMSNorm)


@dataclass
class ExpertTensorSlice:
    """Representa el slice binario de un tensor perteneciente a un único experto.
    
    Permite direccionar y extraer los pesos de un único experto sin obligar
    a cargar ni manipular los restantes expertos de la capa.
    """
    tensor_name: str
    tensor_role: TensorRole
    layer_id: int
    expert_id: int
    total_experts: int
    shape: Tuple[int, ...]
    quant_type: str
    total_tensor_bytes: int
    n_bytes: int
    offset_in_gguf: int

    @property
    def bytes_per_expert(self) -> int:
        return self.n_bytes

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.tensor_name,
            "tensor_role": self.tensor_role.value,
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "total_experts": self.total_experts,
            "shape": list(self.shape),
            "quant_type": self.quant_type,
            "n_bytes": self.n_bytes,
            "offset_in_gguf": self.offset_in_gguf,
        }


@dataclass
class ExpertTensor:
    """Entidad integral que agrupa todos los slices de tensores que componen
    un único experto en una capa específica (Gate + Up + Down projections).
    """
    layer_id: int
    expert_id: int
    gate_slice: Optional[ExpertTensorSlice] = None
    up_slice: Optional[ExpertTensorSlice] = None
    down_slice: Optional[ExpertTensorSlice] = None
    residency_tier: ResidencyTier = ResidencyTier.COLD
    vram_device_ptr: Optional[int] = None
    ram_host_ptr: Optional[int] = None
    is_pinned: bool = False

    @property
    def total_bytes(self) -> int:
        """Suma de bytes de todos los componentes del experto."""
        total = 0
        if self.gate_slice:
            total += self.gate_slice.n_bytes
        if self.up_slice:
            total += self.up_slice.n_bytes
        if self.down_slice:
            total += self.down_slice.n_bytes
        return total

    @property
    def total_mb(self) -> float:
        """Tamaño total del experto en Megabytes."""
        return round(self.total_bytes / (1024 * 1024), 3)

    @property
    def total_kb(self) -> float:
        """Tamaño total del experto en Kilobytes."""
        return round(self.total_bytes / 1024, 2)

    @property
    def is_complete(self) -> bool:
        """Indica si el experto tiene todas sus proyecciones identificadas."""
        return bool(self.gate_slice and self.up_slice and self.down_slice)

    def get_slices(self) -> list[ExpertTensorSlice]:
        """Retorna la lista de slices disponibles en orden [down, gate, up]."""
        slices = []
        if self.down_slice:
            slices.append(self.down_slice)
        if self.gate_slice:
            slices.append(self.gate_slice)
        if self.up_slice:
            slices.append(self.up_slice)
        return slices

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "total_bytes": self.total_bytes,
            "total_mb": self.total_mb,
            "residency_tier": self.residency_tier.value,
            "is_pinned": self.is_pinned,
            "is_complete": self.is_complete,
            "slices": [s.to_dict() for s in self.get_slices()],
        }
