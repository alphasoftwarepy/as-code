"""
AS-Core MoE Engine — Multi-Expert Executor (B4.2)
=================================================
Gestiona y ejecuta el conjunto de 4 expertos activos (Routing Set) en la GPU
manteniendo los 4 expertos simultáneamente en VRAM de forma aislada a los restantes 56.
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gguf
import gguf.quants as gq
import numpy as np

from core.moe.cublas_backend import CuBLASBackend
from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.expert_tensor import ExpertTensor, ResidencyTier
from core.moe.single_expert_executor import SingleExpertExecutionResult
from core.moe.vram_pool import VRAMExpertPool

logger = logging.getLogger("as-code.core.moe.multi_executor")


@dataclass
class MultiExpertExecutionResult:
    """Resultado de la ejecución de los 4 expertos activos de una capa."""
    layer_id: int
    expert_ids: List[int]
    expert_outputs: List[np.ndarray] # Lista de 4 arrays [1, in_dim]
    individual_compute_times_ms: List[float]
    total_gpu_compute_time_ms: float
    total_transfer_time_ms: float
    total_time_ms: float
    total_vram_bytes_allocated: int
    cosine_similarities: List[float]
    max_absolute_errors: List[float]
    is_all_exact: bool


class MultiExpertExecutor:
    """Ejecutor para un conjunto simultáneo de K expertos activos (ej: 4 expertos) en VRAM."""

    def __init__(
        self,
        registry: ExpertRegistry,
        k_active: int = 4,
        cuda_driver: Optional[CUDADriver] = None,
        cublas_backend: Optional[CuBLASBackend] = None,
    ):
        self.registry = registry
        self.k_active = k_active
        self.driver = cuda_driver or CUDADriver()
        self.cublas = cublas_backend or CuBLASBackend(cuda_driver=self.driver)

        self.in_dim = self.registry.profile.embedding_length
        self.hidden_dim = self.registry.profile.expert_feed_forward_length

        # Buffers de activación compartidos en GPU para 1 token
        self._d_x: int = 0
        self._d_gate_out: int = 0
        self._d_up_out: int = 0
        self._d_hidden: int = 0
        self._d_final_out: int = 0

        # Buffers de pesos en VRAM para cada uno de los K slots
        self._d_slots: List[Dict[str, int]] = []
        self._slot_loaded_expert: List[Optional[Tuple[int, int]]] = [None] * self.k_active

        self._init_gpu_buffers()

    def _init_gpu_buffers(self) -> None:
        """Asigna los buffers de activación y los K slots de expertos en VRAM."""
        if not self.driver.is_initialized:
            return

        batch_size = 1
        float_size = 4 # float32

        # 1. Buffers de activación (compartidos y reutilizados secuencialmente por cada experto)
        self._d_x = self.driver.mem_alloc(batch_size * self.in_dim * float_size)
        self._d_gate_out = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_up_out = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_hidden = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_final_out = self.driver.mem_alloc(batch_size * self.in_dim * float_size)

        # 2. Asignar K slots continuos para pesos Float32 en VRAM
        gate_bytes = self.hidden_dim * self.in_dim * float_size
        down_bytes = self.in_dim * self.hidden_dim * float_size

        for i in range(self.k_active):
            slot_ptrs = {
                "w_gate": self.driver.mem_alloc(gate_bytes),
                "w_up": self.driver.mem_alloc(gate_bytes),
                "w_down": self.driver.mem_alloc(down_bytes),
            }
            self._d_slots.append(slot_ptrs)

    def load_routing_set_to_vram(self, layer_id: int, expert_ids: List[int]) -> float:
        """Carga los 4 expertos activos en sus respectivos slots de VRAM."""
        if len(expert_ids) != self.k_active:
            raise ValueError(f"Se esperaban {self.k_active} expertos, recibidos {len(expert_ids)}")

        t0 = time.perf_counter()
        layer_tensors = self.registry.get_layer_raw_tensors(layer_id)
        gate_t = layer_tensors["gate"]
        up_t = layer_tensors["up"]
        down_t = layer_tensors["down"]

        for slot_idx, exp_id in enumerate(expert_ids):
            if self._slot_loaded_expert[slot_idx] == (layer_id, exp_id):
                continue # Ya residente en este slot

            # Extraer y des-cuantizar exclusivamente este experto
            W_gate = gq.dequantize(gate_t.data[exp_id], gate_t.tensor_type).astype(np.float32)
            W_up = gq.dequantize(up_t.data[exp_id], up_t.tensor_type).astype(np.float32)
            W_down = gq.dequantize(down_t.data[exp_id], down_t.tensor_type).astype(np.float32)

            ptrs = self._d_slots[slot_idx]
            self.driver.memcpy_htod(ptrs["w_gate"], W_gate.tobytes(), W_gate.nbytes)
            self.driver.memcpy_htod(ptrs["w_up"], W_up.tobytes(), W_up.nbytes)
            self.driver.memcpy_htod(ptrs["w_down"], W_down.tobytes(), W_down.nbytes)

            self._slot_loaded_expert[slot_idx] = (layer_id, exp_id)

        self.driver.synchronize()
        t_transfer_ms = (time.perf_counter() - t0) * 1000.0
        return t_transfer_ms

    def compute_cpu_reference(self, layer_id: int, expert_id: int, x: np.ndarray) -> np.ndarray:
        """Calcula la referencia CPU para verificación."""
        layer_tensors = self.registry.get_layer_raw_tensors(layer_id)
        gate_t = layer_tensors["gate"]
        up_t = layer_tensors["up"]
        down_t = layer_tensors["down"]

        W_gate = gq.dequantize(gate_t.data[expert_id], gate_t.tensor_type).astype(np.float32)
        W_up = gq.dequantize(up_t.data[expert_id], up_t.tensor_type).astype(np.float32)
        W_down = gq.dequantize(down_t.data[expert_id], down_t.tensor_type).astype(np.float32)

        gate_out = x @ W_gate.T
        up_out = x @ W_up.T
        silu_gate = gate_out / (1.0 + np.exp(-gate_out))
        hidden = silu_gate * up_out
        out = hidden @ W_down.T
        return out

    def execute_routing_set(
        self,
        layer_id: int,
        expert_ids: List[int],
        x: np.ndarray,
        verify_against_cpu_ref: bool = True,
    ) -> MultiExpertExecutionResult:
        """Ejecuta secuencialmente los 4 expertos activos que residen simultáneamente en VRAM."""
        x_fp32 = np.ascontiguousarray(x, dtype=np.float32)
        if x_fp32.ndim == 1:
            x_fp32 = x_fp32.reshape(1, -1)

        batch_size = x_fp32.shape[0]

        # 1. Cargar el set de 4 expertos en VRAM
        t_transfer_ms = self.load_routing_set_to_vram(layer_id, expert_ids)

        # 2. Transferir input vector x a GPU
        self.driver.memcpy_htod(self._d_x, x_fp32.tobytes(), x_fp32.nbytes)

        outputs = []
        compute_times = []
        cos_sims = []
        max_errs = []

        gate_buf = np.empty((batch_size, self.hidden_dim), dtype=np.float32)
        up_buf = np.empty((batch_size, self.hidden_dim), dtype=np.float32)

        for slot_idx, exp_id in enumerate(expert_ids):
            t_exp_0 = time.perf_counter()
            ptrs = self._d_slots[slot_idx]

            # Gate GEMM
            self.cublas.linear_forward_row_major(
                d_x_ptr=self._d_x,
                d_w_ptr=ptrs["w_gate"],
                d_out_ptr=self._d_gate_out,
                batch_size=batch_size,
                in_features=self.in_dim,
                out_features=self.hidden_dim,
            )

            # Up GEMM
            self.cublas.linear_forward_row_major(
                d_x_ptr=self._d_x,
                d_w_ptr=ptrs["w_up"],
                d_out_ptr=self._d_up_out,
                batch_size=batch_size,
                in_features=self.in_dim,
                out_features=self.hidden_dim,
            )
            self.driver.synchronize()

            # SwiGLU activation: silu(gate) * up
            self.driver._lib.cuMemcpyDtoH_v2(gate_buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_gate_out), gate_buf.nbytes)
            self.driver._lib.cuMemcpyDtoH_v2(up_buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_up_out), up_buf.nbytes)

            hidden_buf = (gate_buf / (1.0 + np.exp(-gate_buf))) * up_buf
            self.driver.memcpy_htod(self._d_hidden, hidden_buf.tobytes(), hidden_buf.nbytes)

            # Down GEMM
            self.cublas.linear_forward_row_major(
                d_x_ptr=self._d_hidden,
                d_w_ptr=ptrs["w_down"],
                d_out_ptr=self._d_final_out,
                batch_size=batch_size,
                in_features=self.hidden_dim,
                out_features=self.in_dim,
            )
            self.driver.synchronize()
            t_exp_ms = (time.perf_counter() - t_exp_0) * 1000.0
            compute_times.append(t_exp_ms)

            # Leer salida del experto desde GPU
            out_gpu = np.empty((batch_size, self.in_dim), dtype=np.float32)
            self.driver._lib.cuMemcpyDtoH_v2(out_gpu.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_final_out), out_gpu.nbytes)
            outputs.append(out_gpu)

            # Verificación numérica individual
            if verify_against_cpu_ref:
                out_ref = self.compute_cpu_reference(layer_id, exp_id, x_fp32)
                sim = float(np.dot(out_gpu.flatten(), out_ref.flatten()) / (np.linalg.norm(out_gpu) * np.linalg.norm(out_ref)))
                err = float(np.max(np.abs(out_gpu - out_ref)))
                cos_sims.append(sim)
                max_errs.append(err)
            else:
                cos_sims.append(1.0)
                max_errs.append(0.0)

        total_gpu_compute = sum(compute_times)
        total_time = t_transfer_ms + total_gpu_compute

        # Total VRAM ocupada por los 4 expertos (weights + activaciones)
        single_exp_weights_b = (2 * self.hidden_dim * self.in_dim * 4) + (self.in_dim * self.hidden_dim * 4)
        total_vram_b = (self.k_active * single_exp_weights_b) + (batch_size * (self.in_dim + 3 * self.hidden_dim + self.in_dim) * 4)

        is_all_exact = all(s >= 0.9999 for s in cos_sims) and all(e < 1e-3 for e in max_errs)

        return MultiExpertExecutionResult(
            layer_id=layer_id,
            expert_ids=expert_ids,
            expert_outputs=outputs,
            individual_compute_times_ms=compute_times,
            total_gpu_compute_time_ms=total_gpu_compute,
            total_transfer_time_ms=t_transfer_ms,
            total_time_ms=total_time,
            total_vram_bytes_allocated=total_vram_b,
            cosine_similarities=cos_sims,
            max_absolute_errors=max_errs,
            is_all_exact=is_all_exact,
        )

    def release(self) -> None:
        """Libera todos los buffers de VRAM al terminar."""
        if self.driver.is_initialized:
            # 1. Liberar buffers de activación
            for ptr in [self._d_x, self._d_gate_out, self._d_up_out, self._d_hidden, self._d_final_out]:
                if ptr != 0:
                    try:
                        self.driver.mem_free(ptr)
                    except Exception:
                        pass
            # 2. Liberar K slots de expertos
            for slot in self._d_slots:
                for ptr in slot.values():
                    if ptr != 0:
                        try:
                            self.driver.mem_free(ptr)
                        except Exception:
                            pass
        self._d_slots.clear()
        self._slot_loaded_expert = [None] * self.k_active
        self.cublas.destroy()
        logger.info("[MultiExpertExecutor] Recursos de 4 expertos en GPU liberados.")

    def __del__(self):
        self.release()
