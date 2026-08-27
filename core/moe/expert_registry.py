"""
AS-Core MoE Engine — Expert Registry
====================================
Indexa, organiza y gestiona el acceso a nivel de slice para cada experto individual
de cualquier modelo MoE. Proporciona mapeo determinista de offsets, tamaños y ubicaciones.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import gguf

from core.moe.expert_tensor import (
    ExpertTensor,
    ExpertTensorSlice,
    ResidencyTier,
    TensorRole,
)
from core.moe.model_profile import ModelProfile

logger = logging.getLogger("as-code.core.moe.registry")


class ExpertRegistry:
    """Registro indexado de todos los expertos y sus tensores constituyentes en un modelo MoE."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Archivo GGUF no encontrado: {model_path}")

        # 1. Obtener perfil del modelo
        self.profile = ModelProfile.from_gguf(self.model_path)
        if not self.profile.is_moe:
            raise ValueError(f"El modelo {self.model_path.name} no es una arquitectura MoE (expert_count={self.profile.expert_count})")

        # 2. Mantener GGUFReader persistente para acceso O(1) a mmap data
        self.reader = gguf.GGUFReader(str(self.model_path))

        # 3. Estructura de indexación: layers[layer_id][expert_id] -> ExpertTensor
        self._layers: Dict[int, Dict[int, ExpertTensor]] = {}
        self._dense_tensor_names: List[str] = []
        self._router_tensors: Dict[int, str] = {}
        self._layer_tensors: Dict[int, Dict[str, Any]] = {}

        # 4. Indexar automáticamente
        self._build_index()

    def _determine_role(self, tensor_name: str) -> TensorRole:
        """Determina el rol funcional del tensor a partir de su nombre."""
        name_lower = tensor_name.lower()
        if "gate_exps" in name_lower or "gate_exp" in name_lower:
            return TensorRole.GATE
        elif "up_exps" in name_lower or "up_exp" in name_lower:
            return TensorRole.UP
        elif "down_exps" in name_lower or "down_exp" in name_lower:
            return TensorRole.DOWN
        elif "gate_inp" in name_lower or "router" in name_lower:
            return TensorRole.ROUTER
        elif "attn" in name_lower:
            return TensorRole.ATTENTION
        elif "embd" in name_lower or "output" in name_lower:
            return TensorRole.EMBEDDING
        elif "norm" in name_lower:
            return TensorRole.NORM
        return TensorRole.SHARED

    def _build_index(self) -> None:
        """Lee el archivo GGUF e indexa cada tensor y slice de experto."""
        # Pre-inicializar matriz de capas y expertos
        for l_id in range(self.profile.block_count):
            self._layers[l_id] = {}
            self._layer_tensors[l_id] = {}
            for exp_id in range(self.profile.expert_count):
                self._layers[l_id][exp_id] = ExpertTensor(
                    layer_id=l_id,
                    expert_id=exp_id,
                    residency_tier=ResidencyTier.COLD,
                )

        for tensor in self.reader.tensors:
            t_name = tensor.name
            t_bytes = int(tensor.n_bytes)
            t_offset = int(tensor.data_offset)
            t_shape = tuple(int(s) for s in tensor.shape)
            t_type = str(tensor.tensor_type)

            role = self._determine_role(t_name)

            if role in (TensorRole.GATE, TensorRole.UP, TensorRole.DOWN):
                parts = t_name.split(".")
                layer_id = None
                for part in parts:
                    if part.isdigit():
                        layer_id = int(part)
                        break

                if layer_id is None or layer_id not in self._layers:
                    continue

                if role == TensorRole.GATE:
                    self._layer_tensors[layer_id]["gate"] = tensor
                elif role == TensorRole.UP:
                    self._layer_tensors[layer_id]["up"] = tensor
                elif role == TensorRole.DOWN:
                    self._layer_tensors[layer_id]["down"] = tensor

                num_experts = t_shape[-1]
                bytes_per_exp = t_bytes // num_experts

                for exp_id in range(num_experts):
                    slice_obj = ExpertTensorSlice(
                        tensor_name=t_name,
                        tensor_role=role,
                        layer_id=layer_id,
                        expert_id=exp_id,
                        total_experts=num_experts,
                        shape=t_shape,
                        quant_type=t_type,
                        total_tensor_bytes=t_bytes,
                        n_bytes=bytes_per_exp,
                        offset_in_gguf=t_offset + (exp_id * bytes_per_exp),
                    )

                    expert = self._layers[layer_id][exp_id]
                    if role == TensorRole.GATE:
                        expert.gate_slice = slice_obj
                    elif role == TensorRole.UP:
                        expert.up_slice = slice_obj
                    elif role == TensorRole.DOWN:
                        expert.down_slice = slice_obj

            elif role == TensorRole.ROUTER:
                parts = t_name.split(".")
                for part in parts:
                    if part.isdigit():
                        self._router_tensors[int(part)] = t_name
                        break
                self._dense_tensor_names.append(t_name)
            else:
                self._dense_tensor_names.append(t_name)

        logger.info(
            f"[ExpertRegistry] Indexación completa: {len(self._layers)} capas, "
            f"{self.profile.expert_count} expertos/capa ({self.total_experts_count} expertos totales)."
        )

    @property
    def total_layers(self) -> int:
        return len(self._layers)

    @property
    def total_experts_count(self) -> int:
        return self.profile.block_count * self.profile.expert_count

    def get_layer_raw_tensors(self, layer_id: int) -> Dict[str, Any]:
        """Obtiene las referencias a los tensores raw de la capa desde el reader persistente."""
        return self._layer_tensors.get(layer_id, {})

    def get_expert(self, layer_id: int, expert_id: int) -> ExpertTensor:
        """Obtiene la representación completa de un experto individual."""
        if layer_id not in self._layers:
            raise IndexError(f"Capa {layer_id} fuera de rango (0..{self.profile.block_count-1})")
        if expert_id not in self._layers[layer_id]:
            raise IndexError(f"Experto {expert_id} fuera de rango (0..{self.profile.expert_count-1}) en capa {layer_id}")
        return self._layers[layer_id][expert_id]

    def get_layer_experts(self, layer_id: int) -> Dict[int, ExpertTensor]:
        """Obtiene el diccionario de todos los expertos de una capa específica."""
        if layer_id not in self._layers:
            raise IndexError(f"Capa {layer_id} no encontrada.")
        return self._layers[layer_id]

    def get_routing_set(self, layer_id: int, expert_ids: List[int]) -> List[ExpertTensor]:
        """Obtiene la lista de expertos activados por el router para un token."""
        return [self.get_expert(layer_id, eid) for eid in expert_ids]

    def get_routing_set_bytes(self, layer_id: int, expert_ids: List[int]) -> int:
        """Calcula el peso total en bytes de un set de expertos activados."""
        return sum(self.get_expert(layer_id, eid).total_bytes for eid in expert_ids)

    def get_routing_set_mb(self, layer_id: int, expert_ids: List[int]) -> float:
        """Calcula el peso total en Megabytes de un set de expertos activados."""
        return round(self.get_routing_set_bytes(layer_id, expert_ids) / (1024 * 1024), 3)

    def validate_integrity() -> bool:
        return True

    def validate_integrity(self) -> bool:
        """Verifica que todos los expertos indexados contengan sus 3 proyecciones completas."""
        for l_id, experts in self._layers.items():
            for e_id, exp in experts.items():
                if not exp.is_complete:
                    logger.error(f"Experto incompleto en capa {l_id}, expert {e_id}")
                    return False
        return True

    def get_memory_summary(self) -> Dict[str, Any]:
        """Genera un resumen estructurado del uso de memoria del modelo."""
        sample_exp = self.get_expert(0, 0)
        return {
            "model_name": self.profile.model_name,
            "architecture": self.profile.architecture,
            "total_layers": self.profile.block_count,
            "experts_per_layer": self.profile.expert_count,
            "active_experts_per_token": self.profile.expert_used_count,
            "total_experts_count": self.total_experts_count,
            "single_expert_bytes": sample_exp.total_bytes,
            "single_expert_mb": sample_exp.total_mb,
            "single_expert_kb": sample_exp.total_kb,
            "routing_set_1layer_mb": round((sample_exp.total_bytes * self.profile.expert_used_count) / (1024**2), 2),
            "routing_set_all_layers_mb": round((sample_exp.total_bytes * self.profile.expert_used_count * self.profile.block_count) / (1024**2), 2),
            "dense_memory_mb": self.profile.dense_memory_mb,
            "moe_memory_mb": self.profile.moe_memory_mb,
            "total_model_gb": self.profile.total_file_size_gb,
        }
