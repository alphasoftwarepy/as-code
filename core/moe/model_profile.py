"""
AS-Core MoE Engine — Model Profile Analyzer
============================================
Lee metadatos de modelos GGUF de forma 100% agnóstica al modelo (Qwen, OLMoE, DeepSeek, Gemma MoE, etc.)
y calcula desgloses precisos de memoria, granularidad y topología.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import gguf


def _extract_gguf_value(field_obj: Any) -> Any:
    """Extrae de manera robusta y tipada el valor de un campo GGUF."""
    if field_obj is None or not hasattr(field_obj, "parts") or not field_obj.parts:
        return None

    raw_part = field_obj.parts[-1]
    val_type = field_obj.types[0] if hasattr(field_obj, "types") and field_obj.types else None

    # Strings
    if val_type == gguf.GGUFValueType.STRING or isinstance(raw_part, (bytes, bytearray)):
        try:
            return bytes(raw_part).decode("utf-8", errors="replace")
        except Exception:
            return str(raw_part)

    # Memmap / Arrays
    if hasattr(raw_part, "__len__") and not isinstance(raw_part, (str, bytes)):
        if len(raw_part) == 1:
            item = raw_part[0]
            if hasattr(item, "item"):
                return item.item()
            return int(item) if isinstance(item, (int, float)) and item == int(item) else item
        elif len(raw_part) == 0:
            return None
        else:
            # Lista de elementos
            return [x.item() if hasattr(x, "item") else x for x in raw_part[:20]]

    # Escalar simple
    if hasattr(raw_part, "item"):
        return raw_part.item()

    return raw_part


@dataclass
class ModelProfile:
    """Perfil descriptivo completo de un modelo MoE extraído de sus metadatos GGUF."""
    model_path: str
    model_name: str
    architecture: str
    block_count: int
    expert_count: int
    expert_used_count: int
    embedding_length: int
    expert_feed_forward_length: int
    context_length: int
    quantization: str
    total_file_size_bytes: int
    dense_memory_bytes: int
    moe_memory_bytes: int
    single_expert_bytes: int
    total_tensors_count: int
    is_moe: bool = True

    # ── Métricas Derivadas de Memoria ──────────────────────────

    @property
    def total_file_size_gb(self) -> float:
        return round(self.total_file_size_bytes / (1024**3), 2)

    @property
    def dense_memory_mb(self) -> float:
        return round(self.dense_memory_bytes / (1024**2), 2)

    @property
    def dense_memory_gb(self) -> float:
        return round(self.dense_memory_bytes / (1024**3), 2)

    @property
    def moe_memory_mb(self) -> float:
        return round(self.moe_memory_bytes / (1024**2), 2)

    @property
    def moe_memory_gb(self) -> float:
        return round(self.moe_memory_bytes / (1024**3), 2)

    @property
    def single_expert_mb(self) -> float:
        return round(self.single_expert_bytes / (1024**2), 3)

    @property
    def single_expert_kb(self) -> float:
        return round(self.single_expert_bytes / 1024, 2)

    @property
    def active_experts_layer_mb(self) -> float:
        """Memoria de los expertos activos por token en una sola capa."""
        return round((self.single_expert_bytes * self.expert_used_count) / (1024**2), 2)

    @property
    def active_experts_all_layers_mb(self) -> float:
        """Memoria total de los expertos activos por token a lo largo de todas las capas."""
        return round((self.single_expert_bytes * self.expert_used_count * self.block_count) / (1024**2), 2)

    @property
    def total_experts_in_model(self) -> int:
        return self.block_count * self.expert_count

    def calculate_hotset_capacity(self, available_vram_mb: float, kv_cache_mb: float = 350.0) -> int:
        """Calcula cuántos expertos individuales pueden residir en la VRAM disponible
        después de descontar las atenciones densas y la KV Cache.
        """
        vram_for_experts = available_vram_mb - self.dense_memory_mb - kv_cache_mb
        if vram_for_experts <= 0 or self.single_expert_mb <= 0:
            return 0
        return int(vram_for_experts / self.single_expert_mb)

    def calculate_hotset_per_layer(self, available_vram_mb: float, kv_cache_mb: float = 350.0) -> int:
        """Calcula el número de expertos por capa que pueden colocarse en el HotSet."""
        total_capacity = self.calculate_hotset_capacity(available_vram_mb, kv_cache_mb)
        if self.block_count <= 0:
            return 0
        return min(self.expert_count, total_capacity // self.block_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "is_moe": self.is_moe,
            "block_count": self.block_count,
            "expert_count": self.expert_count,
            "expert_used_count": self.expert_used_count,
            "total_experts_in_model": self.total_experts_in_model,
            "embedding_length": self.embedding_length,
            "expert_feed_forward_length": self.expert_feed_forward_length,
            "context_length": self.context_length,
            "quantization": self.quantization,
            "total_file_size_bytes": self.total_file_size_bytes,
            "total_file_size_gb": self.total_file_size_gb,
            "dense_memory_mb": self.dense_memory_mb,
            "moe_memory_mb": self.moe_memory_mb,
            "single_expert_bytes": self.single_expert_bytes,
            "single_expert_mb": self.single_expert_mb,
            "active_experts_layer_mb": self.active_experts_layer_mb,
            "active_experts_all_layers_mb": self.active_experts_all_layers_mb,
            "total_tensors_count": self.total_tensors_count,
        }

    # ── Factory Constructor Dinámico ───────────────────────────

    @classmethod
    def from_gguf(cls, model_path: Union[str, Path]) -> ModelProfile:
        """Analiza dinámicamente un archivo GGUF y extrae su ModelProfile."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")

        file_size = p.stat().st_size
        reader = gguf.GGUFReader(str(p))

        # 1. Extraer metadatos tipados
        metadata: Dict[str, Any] = {}
        for k, field_obj in reader.fields.items():
            metadata[k] = _extract_gguf_value(field_obj)

        arch = str(metadata.get("general.architecture") or "unknown")
        model_name = str(metadata.get("general.name") or p.stem)
        
        # Parámetros arquitectónicos dinámicos usando el prefijo de la arquitectura
        block_count = int(metadata.get(f"{arch}.block_count") or 0)
        expert_count = int(metadata.get(f"{arch}.expert_count") or 0)
        expert_used_count = int(metadata.get(f"{arch}.expert_used_count") or 0)
        embedding_length = int(metadata.get(f"{arch}.embedding_length") or 0)
        expert_feed_forward_length = int(metadata.get(f"{arch}.expert_feed_forward_length") or 0)
        context_length = int(metadata.get(f"{arch}.context_length") or 2048)
        quantization = str(metadata.get("general.file_type") or "unknown")

        # Determinar si es modelo MoE
        is_moe = expert_count > 0

        # 2. Clasificar tensores y calcular tamaños exactos
        dense_bytes = 0
        moe_bytes = 0
        layer0_moe_bytes = 0

        for tensor in reader.tensors:
            t_name = tensor.name
            t_bytes = int(tensor.n_bytes)
            
            if "exps" in t_name:
                moe_bytes += t_bytes
                if "blk.0." in t_name or ".0." in t_name:
                    layer0_moe_bytes += t_bytes
            else:
                dense_bytes += t_bytes

        # Calcular tamaño de 1 experto
        if expert_count > 0 and layer0_moe_bytes > 0:
            single_expert_bytes = int(layer0_moe_bytes / expert_count)
        elif expert_count > 0 and moe_bytes > 0 and block_count > 0:
            single_expert_bytes = int(moe_bytes / (block_count * expert_count))
        else:
            single_expert_bytes = 0

        return cls(
            model_path=str(p),
            model_name=model_name,
            architecture=arch,
            block_count=block_count,
            expert_count=expert_count,
            expert_used_count=expert_used_count,
            embedding_length=embedding_length,
            expert_feed_forward_length=expert_feed_forward_length,
            context_length=context_length,
            quantization=quantization,
            total_file_size_bytes=file_size,
            dense_memory_bytes=dense_bytes,
            moe_memory_bytes=moe_bytes,
            single_expert_bytes=single_expert_bytes,
            total_tensors_count=len(reader.tensors),
            is_moe=is_moe,
        )
