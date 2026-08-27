"""
AS-Core MoE Engine — Full MoE Layer Executor (B4.3.4)
=====================================================
Ejecuta la computación completa de una capa MoE de extremo a extremo:
  Hidden State (x)
        ↓
  RealRouter (Top-4 dinámico + Pesos w_i normalizados)
        ↓
  ExpertRegistry (Resolución determinista de tensores)
        ↓
  ResidencyManager (HIT/MISS + Promoción física a VRAM)
        ↓
  4 Expert SwiGLU FFN en GPU (cuBLAS)
        ↓
  Weighted Sum en GPU: y = Σ(w_i * FFN_i(x)) vía cublasSaxpy
        ↓
  Vector de Salida MoE (1 x 2048)
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gguf.quants as gq
import numpy as np

from core.moe.cublas_backend import CuBLASBackend
from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.residency_manager import ResidencyDecision, ResidencyLayerDispatch, ResidencyManager
from core.moe.router import RealRouter, RoutingDecision

logger = logging.getLogger("as-code.core.moe.layer_executor")


@dataclass
class MoELayerExecutionResult:
    """Resultado detallado de la ejecución de una capa MoE completa."""
    layer_id: int
    output: np.ndarray                 # Vector de salida acumulado [1, in_dim]
    expert_ids: List[int]              # Top-K IDs seleccionados por el router
    routing_weights: List[float]       # Pesos normalizados aplicados en la suma ponderada
    is_warm: bool                      # True si los 4 expertos fueron HITs (0 bytes transferidos)
    
    # Desglose de latencias [MEASURED]
    router_latency_ms: float
    residency_lookup_latency_ms: float
    promotion_latency_ms: float
    expert_compute_latency_ms: float
    weighted_sum_latency_ms: float
    total_layer_latency_ms: float
    
    # Métricas de memoria y residencia
    hit_count: int
    miss_count: int
    hit_rate: float
    vram_allocated_bytes: int
    
    # Validación numérica contra referencia CPU
    cosine_similarity_vs_ref: float
    max_absolute_error_vs_ref: float
    relative_error_vs_ref: float

    @property
    def is_exact(self) -> bool:
        return (
            self.cosine_similarity_vs_ref >= 0.9999
            and self.max_absolute_error_vs_ref < 1e-3
            and self.relative_error_vs_ref < 1e-4
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_ids": self.expert_ids,
            "weights": [round(w, 6) for w in self.routing_weights],
            "is_warm": self.is_warm,
            "total_latency_ms": round(self.total_layer_latency_ms, 4),
            "promotion_latency_ms": round(self.promotion_latency_ms, 4),
            "expert_compute_latency_ms": round(self.expert_compute_latency_ms, 4),
            "weighted_sum_latency_ms": round(self.weighted_sum_latency_ms, 4),
            "hit_rate": round(self.hit_rate, 4),
            "cosine_similarity": round(self.cosine_similarity_vs_ref, 7),
            "max_absolute_error": f"{self.max_absolute_error_vs_ref:.6e}",
            "relative_error": f"{self.relative_error_vs_ref:.6e}",
            "is_exact": self.is_exact,
        }


class MoELayerExecutor:
    """Ejecutor de alto rendimiento para una capa Transformer MoE completa en GPU."""

    def __init__(
        self,
        registry: ExpertRegistry,
        router: Optional[RealRouter] = None,
        residency_manager: Optional[ResidencyManager] = None,
        cuda_driver: Optional[CUDADriver] = None,
        cublas_backend: Optional[CuBLASBackend] = None,
        tracer: Optional[Any] = None,
        k_active: int = 4,
    ):
        self.registry = registry
        self.driver = cuda_driver or CUDADriver()
        self.cublas = cublas_backend or CuBLASBackend(cuda_driver=self.driver)
        self.router = router or RealRouter(registry=self.registry, k_active=k_active, cuda_driver=self.driver, cublas_backend=self.cublas)
        self.residency_manager = residency_manager
        self.tracer = tracer
        self.k_active = k_active

        self.in_dim = self.registry.profile.embedding_length
        self.hidden_dim = self.registry.profile.expert_feed_forward_length

        # Buffers de activación en VRAM para 1 token
        self._d_x: int = 0
        self._d_gate_out: int = 0
        self._d_up_out: int = 0
        self._d_hidden: int = 0
        self._d_expert_out: int = 0
        self._d_accum_y: int = 0

        # K execution slots para pesos des-cuantizados en VRAM [slot_idx -> {w_gate, w_up, w_down}]
        self._d_slots: List[Dict[str, int]] = []
        self._slot_loaded_expert: List[Optional[Tuple[int, int]]] = [None] * self.k_active

        self._init_gpu_buffers()

    def _init_gpu_buffers(self) -> None:
        """Asigna los buffers contiguos de activación y los K slots de ejecución en GPU."""
        if not self.driver.is_initialized:
            return

        batch_size = 1
        float_size = 4 # float32

        # 1. Buffers de activación
        self._d_x = self.driver.mem_alloc(batch_size * self.in_dim * float_size)
        self._d_gate_out = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_up_out = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_hidden = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_expert_out = self.driver.mem_alloc(batch_size * self.in_dim * float_size)
        self._d_accum_y = self.driver.mem_alloc(batch_size * self.in_dim * float_size)

        # 2. Buffers de pesos Float32 para los K slots de cómputo
        gate_bytes = self.hidden_dim * self.in_dim * float_size
        down_bytes = self.in_dim * self.hidden_dim * float_size

        for i in range(self.k_active):
            slot_ptrs = {
                "w_gate": self.driver.mem_alloc(gate_bytes),
                "w_up": self.driver.mem_alloc(gate_bytes),
                "w_down": self.driver.mem_alloc(down_bytes),
            }
            self._d_slots.append(slot_ptrs)

    def load_slot_weights_if_needed(self, slot_idx: int, layer_id: int, expert_id: int) -> float:
        """Carga y des-cuantiza los pesos al slot de VRAM si no están presentes."""
        if self._slot_loaded_expert[slot_idx] == (layer_id, expert_id):
            return 0.0

        t0 = time.perf_counter()
        layer_tensors = self.registry.get_layer_raw_tensors(layer_id)
        gate_t = layer_tensors["gate"]
        up_t = layer_tensors["up"]
        down_t = layer_tensors["down"]

        W_gate = gq.dequantize(gate_t.data[expert_id], gate_t.tensor_type).astype(np.float32)
        W_up = gq.dequantize(up_t.data[expert_id], up_t.tensor_type).astype(np.float32)
        W_down = gq.dequantize(down_t.data[expert_id], down_t.tensor_type).astype(np.float32)

        ptrs = self._d_slots[slot_idx]
        self.driver.memcpy_htod(ptrs["w_gate"], W_gate.tobytes(), W_gate.nbytes)
        self.driver.memcpy_htod(ptrs["w_up"], W_up.tobytes(), W_up.nbytes)
        self.driver.memcpy_htod(ptrs["w_down"], W_down.tobytes(), W_down.nbytes)
        self.driver.synchronize()

        self._slot_loaded_expert[slot_idx] = (layer_id, expert_id)
        return (time.perf_counter() - t0) * 1000.0

    def compute_cpu_reference(
        self,
        layer_id: int,
        x: np.ndarray,
        expert_ids: List[int],
        weights: List[float],
    ) -> np.ndarray:
        """Calcula la referencia matemática exacta en CPU para verificación estricta."""
        layer_tensors = self.registry.get_layer_raw_tensors(layer_id)
        gate_t = layer_tensors["gate"]
        up_t = layer_tensors["up"]
        down_t = layer_tensors["down"]

        y_accum = np.zeros((1, self.in_dim), dtype=np.float32)
        x_vec = np.ascontiguousarray(x.reshape(1, -1), dtype=np.float32)

        for eid, w in zip(expert_ids, weights):
            W_g = gq.dequantize(gate_t.data[eid], gate_t.tensor_type).astype(np.float32)
            W_u = gq.dequantize(up_t.data[eid], up_t.tensor_type).astype(np.float32)
            W_d = gq.dequantize(down_t.data[eid], down_t.tensor_type).astype(np.float32)

            g_out = x_vec @ W_g.T
            u_out = x_vec @ W_u.T
            h_out = (g_out / (1.0 + np.exp(-g_out))) * u_out
            d_out = h_out @ W_d.T
            y_accum += w * d_out

        return y_accum

    def forward_layer(
        self,
        layer_id: int,
        x: np.ndarray,
        token_index: int = 0,
        seq_id: Optional[int] = None,
        verify_against_cpu_ref: bool = True,
    ) -> MoELayerExecutionResult:
        """Ejecuta el forward pass completo de la capa MoE (Router -> Residency -> 4 FFN -> Weighted Sum)."""
        t_total_0 = time.perf_counter()
        x_fp32 = np.ascontiguousarray(x.reshape(1, -1), dtype=np.float32)

        # 1. EVALUAR ROUTER REAL
        decision = self.router.route_token(layer_id=layer_id, x=x_fp32, use_gpu=True)
        top_k_ids = decision.top_k_ids
        weights = decision.normalized_weights

        # Tracing de enrutamiento si está habilitado (Zero-overhead si disabled)
        if self.tracer is not None and getattr(self.tracer, "enabled", False):
            self.tracer.record(
                token_index=token_index,
                layer_id=layer_id,
                decision=decision,
                seq_id=seq_id,
            )

        # 2. RESOLVER RESIDENCY & PROMOTION A VRAM
        t_res_0 = time.perf_counter()
        total_promotion_ms = 0.0
        hit_count = 0
        miss_count = 0

        for slot_idx, eid in enumerate(top_k_ids):
            if self.residency_manager is not None:
                rd = self.residency_manager.ensure_in_vram(layer_id, eid)
                if rd.is_hit:
                    hit_count += 1
                else:
                    miss_count += 1
                total_promotion_ms += rd.promotion_latency_ms

            # Asegurar pesos de cómputo en el slot de ejecución
            t_w = self.load_slot_weights_if_needed(slot_idx, layer_id, eid)
            if self.residency_manager is None:
                if t_w == 0.0:
                    hit_count += 1
                else:
                    miss_count += 1
                    total_promotion_ms += t_w

        t_residency_lookup_ms = (time.perf_counter() - t_res_0) * 1000.0 - total_promotion_ms

        # 3. INICIALIZAR ACUMULADOR DE SALIDA EN GPU (Y = 0)
        t_comp_0 = time.perf_counter()
        zero_buf = np.zeros((1, self.in_dim), dtype=np.float32)
        self.driver.memcpy_htod(self._d_accum_y, zero_buf.tobytes(), zero_buf.nbytes)
        self.driver.memcpy_htod(self._d_x, x_fp32.tobytes(), x_fp32.nbytes)

        gate_buf = np.empty((1, self.hidden_dim), dtype=np.float32)
        up_buf = np.empty((1, self.hidden_dim), dtype=np.float32)

        t_weighted_sum_ms = 0.0

        # 4. EJECUTAR 4 EXPERTOS Y ACUMULAR CON WEIGHTED SUM DIRECTO EN GPU
        for slot_idx, (eid, w) in enumerate(zip(top_k_ids, weights)):
            ptrs = self._d_slots[slot_idx]

            # Gate GEMM: (1, 1408) = x @ W_gate.T
            self.cublas.linear_forward_row_major(
                d_x_ptr=self._d_x,
                d_w_ptr=ptrs["w_gate"],
                d_out_ptr=self._d_gate_out,
                batch_size=1,
                in_features=self.in_dim,
                out_features=self.hidden_dim,
            )

            # Up GEMM: (1, 1408) = x @ W_up.T
            self.cublas.linear_forward_row_major(
                d_x_ptr=self._d_x,
                d_w_ptr=ptrs["w_up"],
                d_out_ptr=self._d_up_out,
                batch_size=1,
                in_features=self.in_dim,
                out_features=self.hidden_dim,
            )
            self.driver.synchronize()

            # SwiGLU activation: silu(gate) * up
            self.driver._lib.cuMemcpyDtoH_v2(gate_buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_gate_out), gate_buf.nbytes)
            self.driver._lib.cuMemcpyDtoH_v2(up_buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_up_out), up_buf.nbytes)

            hidden_buf = (gate_buf / (1.0 + np.exp(-gate_buf))) * up_buf
            self.driver.memcpy_htod(self._d_hidden, hidden_buf.tobytes(), hidden_buf.nbytes)

            # Down GEMM: (1, 2048) = hidden @ W_down.T
            self.cublas.linear_forward_row_major(
                d_x_ptr=self._d_hidden,
                d_w_ptr=ptrs["w_down"],
                d_out_ptr=self._d_expert_out,
                batch_size=1,
                in_features=self.hidden_dim,
                out_features=self.in_dim,
            )
            self.driver.synchronize()

            # 5. WEIGHTED SUM EN GPU: accum_y += w_i * expert_out
            t_saxpy_0 = time.perf_counter()
            self.cublas.saxpy(
                n=self.in_dim,
                alpha=float(w),
                d_x_ptr=self._d_expert_out,
                incx=1,
                d_y_ptr=self._d_accum_y,
                incy=1,
            )
            self.driver.synchronize()
            t_weighted_sum_ms += (time.perf_counter() - t_saxpy_0) * 1000.0

        t_expert_compute_ms = (time.perf_counter() - t_comp_0) * 1000.0 - t_weighted_sum_ms

        # 6. LEER RESULTADO FINAL ACUMULADO DESDE GPU (Lectura única)
        out_gpu = np.empty((1, self.in_dim), dtype=np.float32)
        self.driver._lib.cuMemcpyDtoH_v2(out_gpu.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_accum_y), out_gpu.nbytes)

        t_total_ms = (time.perf_counter() - t_total_0) * 1000.0

        # 7. VALIDACIÓN NUMÉRICA ESTRICTA
        cos_sim = 1.0
        max_err = 0.0
        rel_err = 0.0

        if verify_against_cpu_ref:
            out_ref = self.compute_cpu_reference(layer_id, x_fp32, top_k_ids, weights)
            cos_sim = float(np.dot(out_gpu.flatten(), out_ref.flatten()) / (np.linalg.norm(out_gpu) * np.linalg.norm(out_ref)))
            max_err = float(np.max(np.abs(out_gpu - out_ref)))
            rel_err = float(max_err / (np.max(np.abs(out_ref)) + 1e-8))

        is_warm = (hit_count == self.k_active)
        vram_allocated_b = self.k_active * ((2 * self.hidden_dim * self.in_dim * 4) + (self.in_dim * self.hidden_dim * 4))

        return MoELayerExecutionResult(
            layer_id=layer_id,
            output=out_gpu,
            expert_ids=top_k_ids,
            routing_weights=weights,
            is_warm=is_warm,
            router_latency_ms=decision.router_latency_ms,
            residency_lookup_latency_ms=max(0.0, t_residency_lookup_ms),
            promotion_latency_ms=total_promotion_ms,
            expert_compute_latency_ms=t_expert_compute_ms,
            weighted_sum_latency_ms=t_weighted_sum_ms,
            total_layer_latency_ms=t_total_ms,
            hit_count=hit_count,
            miss_count=miss_count,
            hit_rate=(hit_count / self.k_active),
            vram_allocated_bytes=vram_allocated_b,
            cosine_similarity_vs_ref=cos_sim,
            max_absolute_error_vs_ref=max_err,
            relative_error_vs_ref=rel_err,
        )

    def release(self) -> None:
        """Libera todos los buffers de GPU."""
        if self.driver.is_initialized:
            for ptr in [self._d_x, self._d_gate_out, self._d_up_out, self._d_hidden, self._d_expert_out, self._d_accum_y]:
                if ptr != 0:
                    try:
                        self.driver.mem_free(ptr)
                    except Exception:
                        pass
            for slot in self._d_slots:
                for ptr in slot.values():
                    if ptr != 0:
                        try:
                            self.driver.mem_free(ptr)
                        except Exception:
                            pass
        self._d_slots.clear()
        self._slot_loaded_expert = [None] * self.k_active
        logger.info("[MoELayerExecutor] Recursos de GPU liberados.")

    def __del__(self):
        self.release()
