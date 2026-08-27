"""
AS-Core MoE Engine — Real Router & Top-K Dispatcher (B4.3.1 + B4.3.2)
=====================================================================
Implementa la evaluación del Router de la capa Transformer MoE y su resolución
automática y determinista hacia el ExpertRegistry:
- Proyección lineal de logits: logits = x @ W_gate_inp.T
- Softmax numéricamente estable sobre los N_expert logits
- Extracción Top-K de expertos activos y renormalización estricta de pesos (sum = 1.0)
- Resolución O(1) de Expert IDs hacia ExpertTensor en ExpertRegistry (B4.3.2)
- Verificación estricta de integridad de tensores, slices y cuantización
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gguf
import numpy as np

from core.moe.cublas_backend import CuBLASBackend
from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.expert_tensor import ExpertTensor

logger = logging.getLogger("as-code.core.moe.router")


@dataclass
class RoutingDecision:
    """Decisión de enrutamiento pura para un token en una capa específica (B4.3.1)."""
    layer_id: int
    top_k_ids: List[int]
    raw_probabilities: np.ndarray        # Probabilidades softmax sobre todos los expertos (60)
    normalized_weights: List[float]      # Pesos normalizados de los K seleccionados (sum = 1.0)
    router_latency_ms: float
    is_gpu_accelerated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "top_k_ids": self.top_k_ids,
            "normalized_weights": [round(w, 6) for w in self.normalized_weights],
            "router_latency_ms": round(self.router_latency_ms, 4),
            "is_gpu_accelerated": self.is_gpu_accelerated,
        }


@dataclass
class RoutedExpert:
    """Representa un experto individual activado por el router con sus tensores y peso asignado (B4.3.2)."""
    expert_id: int
    layer_id: int
    weight: float
    probability: float
    tensor: ExpertTensor

    @property
    def total_bytes(self) -> int:
        return self.tensor.total_bytes

    @property
    def total_mb(self) -> float:
        return self.tensor.total_mb


@dataclass
class RoutedLayerDispatch:
    """Resultado del enrutamiento y resolución completa de tensores para una capa (B4.3.2)."""
    layer_id: int
    decision: RoutingDecision
    routed_experts: List[RoutedExpert] # Lista de K expertos resueltos
    total_bytes: int
    total_mb: float
    resolution_latency_ms: float

    @property
    def expert_ids(self) -> List[int]:
        return [re.expert_id for re in self.routed_experts]

    @property
    def weights(self) -> List[float]:
        return [re.weight for re in self.routed_experts]

    @property
    def is_valid(self) -> bool:
        """Verifica que todos los expertos pertenezcan a la capa y tengan sus 3 proyecciones."""
        if len(self.routed_experts) != len(self.decision.top_k_ids):
            return False
        for re in self.routed_experts:
            if re.layer_id != self.layer_id or re.expert_id not in self.decision.top_k_ids:
                return False
            if not re.tensor.is_complete:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_ids": self.expert_ids,
            "weights": [round(w, 6) for w in self.weights],
            "total_bytes": self.total_bytes,
            "total_mb": self.total_mb,
            "router_latency_ms": round(self.decision.router_latency_ms, 4),
            "resolution_latency_ms": round(self.resolution_latency_ms, 4),
            "is_gpu": self.decision.is_gpu_accelerated,
        }


class RealRouter:
    """Router MoE para evaluación de proyección lineal, selección Top-K y resolución en Registry."""

    def __init__(
        self,
        registry: ExpertRegistry,
        k_active: Optional[int] = None,
        cuda_driver: Optional[CUDADriver] = None,
        cublas_backend: Optional[CuBLASBackend] = None,
    ):
        self.registry = registry
        self.k_active = k_active or self.registry.profile.expert_used_count
        self.expert_count = self.registry.profile.expert_count
        self.in_dim = self.registry.profile.embedding_length

        self.driver = cuda_driver or CUDADriver()
        self.cublas = cublas_backend or CuBLASBackend(cuda_driver=self.driver)

        # Cache de matrices de pesos de router en memoria Host [layer_id -> ndarray(60, 2048)]
        self._host_router_weights: Dict[int, np.ndarray] = {}

        # Buffers GPU para router [layer_id -> d_ptr]
        self._d_router_weights: Dict[int, int] = {}
        self._d_x: int = 0
        self._d_logits: int = 0

        self._load_and_init_routers()

    def _load_and_init_routers(self) -> None:
        """Carga los pesos de los routers de todas las capas e inicializa buffers en GPU."""
        for tensor in self.registry.reader.tensors:
            if "gate_inp.weight" in tensor.name and "shexp" not in tensor.name:
                parts = tensor.name.split(".")
                layer_id = None
                for p in parts:
                    if p.isdigit():
                        layer_id = int(p)
                        break
                if layer_id is not None:
                    # En GGUF float32 tensor.data shape es (60, 2048)
                    w_data = np.ascontiguousarray(tensor.data, dtype=np.float32)
                    self._host_router_weights[layer_id] = w_data

        if self.driver.is_initialized and self.cublas.is_available:
            batch_size = 1
            float_size = 4
            self._d_x = self.driver.mem_alloc(batch_size * self.in_dim * float_size)
            self._d_logits = self.driver.mem_alloc(batch_size * self.expert_count * float_size)

            for l_id, w in self._host_router_weights.items():
                d_w = self.driver.mem_alloc(w.nbytes)
                self.driver.memcpy_htod(d_w, w.tobytes(), w.nbytes)
                self._d_router_weights[l_id] = d_w

            logger.info(
                f"[RealRouter] {len(self._d_router_weights)} matrices de Router cargadas en VRAM "
                f"({len(self._d_router_weights) * 480 / 1024:.2f} MB VRAM total)."
            )

    def route_cpu_reference(self, layer_id: int, x: np.ndarray) -> Tuple[List[int], List[float], np.ndarray]:
        """Calcula el enrutamiento de referencia en CPU usando álgebra NumPy estándar."""
        if layer_id not in self._host_router_weights:
            raise KeyError(f"Pesos de router no encontrados para capa {layer_id}")

        W = self._host_router_weights[layer_id] # (60, 2048)
        x_vec = np.ascontiguousarray(x.reshape(-1), dtype=np.float32) # (2048,)

        # 1. Logits = x @ W.T
        logits = x_vec @ W.T # (60,)

        # 2. Stable Softmax
        max_l = np.max(logits)
        exp_l = np.exp(logits - max_l)
        probs = exp_l / np.sum(exp_l)

        # 3. Top-K Selection
        top_k_indices = [int(i) for i in np.argsort(probs)[-self.k_active:][::-1]]
        top_k_probs = probs[top_k_indices]

        # 4. Renormalización
        norm_weights = [float(w) for w in (top_k_probs / np.sum(top_k_probs))]

        return top_k_indices, norm_weights, probs

    def route_token(
        self,
        layer_id: int,
        x: np.ndarray,
        use_gpu: bool = True,
    ) -> RoutingDecision:
        """Evalúa el router para un token y devuelve la decisión de enrutamiento pura (B4.3.1)."""
        x_vec = np.ascontiguousarray(x.reshape(1, -1), dtype=np.float32)
        t0 = time.perf_counter()

        if use_gpu and self.cublas.is_available and layer_id in self._d_router_weights:
            d_w = self._d_router_weights[layer_id]
            self.driver.memcpy_htod(self._d_x, x_vec.tobytes(), x_vec.nbytes)

            self.cublas.linear_forward_row_major(
                d_x_ptr=self._d_x,
                d_w_ptr=d_w,
                d_out_ptr=self._d_logits,
                batch_size=1,
                in_features=self.in_dim,
                out_features=self.expert_count,
            )
            self.driver.synchronize()

            logits = np.empty((self.expert_count,), dtype=np.float32)
            self.driver._lib.cuMemcpyDtoH_v2(
                logits.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint64(self._d_logits),
                logits.nbytes,
            )

            max_l = np.max(logits)
            exp_l = np.exp(logits - max_l)
            probs = exp_l / np.sum(exp_l)
            top_k_indices = [int(i) for i in np.argsort(probs)[-self.k_active:][::-1]]
            top_k_probs = probs[top_k_indices]
            norm_weights = [float(w) for w in (top_k_probs / np.sum(top_k_probs))]
            is_gpu = True
        else:
            top_k_indices, norm_weights, probs = self.route_cpu_reference(layer_id, x_vec)
            is_gpu = False

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RoutingDecision(
            layer_id=layer_id,
            top_k_ids=top_k_indices,
            raw_probabilities=probs,
            normalized_weights=norm_weights,
            router_latency_ms=latency_ms,
            is_gpu_accelerated=is_gpu,
        )

    def route_and_resolve(
        self,
        layer_id: int,
        x: np.ndarray,
        use_gpu: bool = True,
    ) -> RoutedLayerDispatch:
        """[B4.3.2] Evalúa el Router y resuelve de forma O(1) cada ID contra el ExpertRegistry."""
        t_res_0 = time.perf_counter()

        # 1. Obtener decisión del Router
        decision = self.route_token(layer_id=layer_id, x=x, use_gpu=use_gpu)

        # 2. Resolver cada ID en ExpertRegistry
        routed_experts: List[RoutedExpert] = []
        total_bytes = 0

        for eid, w in zip(decision.top_k_ids, decision.normalized_weights):
            prob = float(decision.raw_probabilities[eid])
            expert_tensor = self.registry.get_expert(layer_id, eid)

            # Validación de identidad e integridad
            if expert_tensor.layer_id != layer_id:
                raise ValueError(f"Discrepancia de Capa: esperada {layer_id}, obtenida {expert_tensor.layer_id}")
            if expert_tensor.expert_id != eid:
                raise ValueError(f"Discrepancia de Experto: esperado {eid}, obtenido {expert_tensor.expert_id}")
            if not expert_tensor.is_complete:
                raise ValueError(f"Experto incompleto: ({layer_id}, {eid})")

            routed_exp = RoutedExpert(
                expert_id=eid,
                layer_id=layer_id,
                weight=w,
                probability=prob,
                tensor=expert_tensor,
            )
            routed_experts.append(routed_exp)
            total_bytes += expert_tensor.total_bytes

        t_resolution_ms = (time.perf_counter() - t_res_0) * 1000.0

        return RoutedLayerDispatch(
            layer_id=layer_id,
            decision=decision,
            routed_experts=routed_experts,
            total_bytes=total_bytes,
            total_mb=round(total_bytes / (1024 * 1024), 3),
            resolution_latency_ms=t_resolution_ms,
        )

    def release(self) -> None:
        """Libera los buffers de GPU."""
        if self.driver.is_initialized:
            for ptr in [self._d_x, self._d_logits]:
                if ptr != 0:
                    try:
                        self.driver.mem_free(ptr)
                    except Exception:
                        pass
            for ptr in self._d_router_weights.values():
                if ptr != 0:
                    try:
                        self.driver.mem_free(ptr)
                    except Exception:
                        pass
            self._d_router_weights.clear()
        logger.info("[RealRouter] Recursos de GPU liberados.")

    def __del__(self):
        self.release()
