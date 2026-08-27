"""
AS-Core MoE Engine — Single Expert Executor (B4.1)
==================================================
Ejecuta de forma aislada y granular la computación SwiGLU FFN de un único experto MoE
en la GPU mediante cuBLAS y punteros de VRAM sin cargar los restantes 59 expertos.
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import gguf
import gguf.quants as gq
import numpy as np

from core.moe.cublas_backend import CuBLASBackend
from core.moe.cuda_driver import CUDADriver
from core.moe.expert_registry import ExpertRegistry
from core.moe.expert_tensor import ExpertTensor, ResidencyTier
from core.moe.vram_pool import VRAMExpertPool

logger = logging.getLogger("as-code.core.moe.executor")


@dataclass
class SingleExpertExecutionResult:
    """Resultado de la ejecución de inferencia sobre un único experto."""
    layer_id: int
    expert_id: int
    output: np.ndarray
    compute_time_ms: float
    transfer_time_ms: float
    total_time_ms: float
    vram_bytes_used: int
    cosine_similarity_vs_ref: float
    max_absolute_error_vs_ref: float

    @property
    def is_numerically_exact(self) -> bool:
        return self.cosine_similarity_vs_ref >= 0.9999 and self.max_absolute_error_vs_ref < 1e-3


class SingleExpertExecutor:
    """Ejecutor de bajo nivel para un único experto MoE sobre GPU."""

    def __init__(
        self,
        registry: ExpertRegistry,
        vram_pool: Optional[VRAMExpertPool] = None,
        cuda_driver: Optional[CUDADriver] = None,
        cublas_backend: Optional[CuBLASBackend] = None,
    ):
        self.registry = registry
        self.driver = cuda_driver or CUDADriver()
        self.cublas = cublas_backend or CuBLASBackend(cuda_driver=self.driver)

        self.in_dim = self.registry.profile.embedding_length
        self.hidden_dim = self.registry.profile.expert_feed_forward_length
        self.vram_pool = vram_pool

        # Buffers temporales de activación en VRAM para 1 token
        self._d_x: int = 0
        self._d_gate_out: int = 0
        self._d_up_out: int = 0
        self._d_hidden: int = 0
        self._d_final_out: int = 0

        # Buffers de pesos del experto en VRAM
        self._d_w_gate: int = 0
        self._d_w_up: int = 0
        self._d_w_down: int = 0
        self._current_loaded_expert: Optional[Tuple[int, int]] = None

        self._init_gpu_buffers()

    def _init_gpu_buffers(self) -> None:
        """Asigna los buffers contiguos de activación en GPU."""
        if not self.driver.is_initialized:
            return

        batch_size = 1
        float_size = 4 # float32

        # 1. Buffers de activación
        self._d_x = self.driver.mem_alloc(batch_size * self.in_dim * float_size)
        self._d_gate_out = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_up_out = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_hidden = self.driver.mem_alloc(batch_size * self.hidden_dim * float_size)
        self._d_final_out = self.driver.mem_alloc(batch_size * self.in_dim * float_size)

        # 2. Buffers de pesos Float32 para 1 experto
        gate_bytes = self.hidden_dim * self.in_dim * float_size
        down_bytes = self.in_dim * self.hidden_dim * float_size

        self._d_w_gate = self.driver.mem_alloc(gate_bytes)
        self._d_w_up = self.driver.mem_alloc(gate_bytes)
        self._d_w_down = self.driver.mem_alloc(down_bytes)

    def load_expert_weights_to_gpu(self, layer_id: int, expert_id: int) -> float:
        """Carga y des-cuantiza los pesos de un único experto en VRAM de forma ultra rápida."""
        if self._current_loaded_expert == (layer_id, expert_id):
            return 0.0

        t0 = time.perf_counter()
        layer_tensors = self.registry.get_layer_raw_tensors(layer_id)
        gate_t = layer_tensors["gate"]
        up_t = layer_tensors["up"]
        down_t = layer_tensors["down"]

        # Extraer exclusivamente el slice del experto solicitado desde memmap
        W_gate = gq.dequantize(gate_t.data[expert_id], gate_t.tensor_type).astype(np.float32)
        W_up = gq.dequantize(up_t.data[expert_id], up_t.tensor_type).astype(np.float32)
        W_down = gq.dequantize(down_t.data[expert_id], down_t.tensor_type).astype(np.float32)

        # Transferir a VRAM
        self.driver.memcpy_htod(self._d_w_gate, W_gate.tobytes(), W_gate.nbytes)
        self.driver.memcpy_htod(self._d_w_up, W_up.tobytes(), W_up.nbytes)
        self.driver.memcpy_htod(self._d_w_down, W_down.tobytes(), W_down.nbytes)
        self.driver.synchronize()

        self._current_loaded_expert = (layer_id, expert_id)
        t_transfer_ms = (time.perf_counter() - t0) * 1000.0
        return t_transfer_ms

    def compute_cpu_reference(self, layer_id: int, expert_id: int, x: np.ndarray) -> np.ndarray:
        """Calcula la referencia matemática exacta en CPU para verificación de tolerancia."""
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

    def execute_single_expert(
        self,
        layer_id: int,
        expert_id: int,
        x: np.ndarray,
        verify_against_cpu_ref: bool = True,
    ) -> SingleExpertExecutionResult:
        """Ejecuta la inferencia SwiGLU FFN de un único experto en la GPU real."""
        x_fp32 = np.ascontiguousarray(x, dtype=np.float32)
        if x_fp32.ndim == 1:
            x_fp32 = x_fp32.reshape(1, -1)

        batch_size = x_fp32.shape[0]

        # 1. Transferir pesos a GPU si no están presentes
        t_transfer_ms = self.load_expert_weights_to_gpu(layer_id, expert_id)

        # 2. Transferir input vector x a GPU
        self.driver.memcpy_htod(self._d_x, x_fp32.tobytes(), x_fp32.nbytes)

        t_comp_0 = time.perf_counter()

        # 3. GEMM Gate: (batch, hidden_dim) = x @ W_gate.T
        self.cublas.linear_forward_row_major(
            d_x_ptr=self._d_x,
            d_w_ptr=self._d_w_gate,
            d_out_ptr=self._d_gate_out,
            batch_size=batch_size,
            in_features=self.in_dim,
            out_features=self.hidden_dim,
        )

        # 4. GEMM Up: (batch, hidden_dim) = x @ W_up.T
        self.cublas.linear_forward_row_major(
            d_x_ptr=self._d_x,
            d_w_ptr=self._d_w_up,
            d_out_ptr=self._d_up_out,
            batch_size=batch_size,
            in_features=self.in_dim,
            out_features=self.hidden_dim,
        )
        self.driver.synchronize()

        # 5. Activación SwiGLU: silu(gate) * up
        gate_buf = np.empty((batch_size, self.hidden_dim), dtype=np.float32)
        up_buf = np.empty((batch_size, self.hidden_dim), dtype=np.float32)

        self.driver._lib.cuMemcpyDtoH_v2(gate_buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_gate_out), gate_buf.nbytes)
        self.driver._lib.cuMemcpyDtoH_v2(up_buf.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_up_out), up_buf.nbytes)

        hidden_buf = (gate_buf / (1.0 + np.exp(-gate_buf))) * up_buf
        self.driver.memcpy_htod(self._d_hidden, hidden_buf.tobytes(), hidden_buf.nbytes)

        # 6. GEMM Down: (batch, in_dim) = hidden @ W_down.T
        self.cublas.linear_forward_row_major(
            d_x_ptr=self._d_hidden,
            d_w_ptr=self._d_w_down,
            d_out_ptr=self._d_final_out,
            batch_size=batch_size,
            in_features=self.hidden_dim,
            out_features=self.in_dim,
        )
        self.driver.synchronize()
        t_compute_ms = (time.perf_counter() - t_comp_0) * 1000.0

        # 7. Leer resultado final desde GPU
        out_gpu = np.empty((batch_size, self.in_dim), dtype=np.float32)
        self.driver._lib.cuMemcpyDtoH_v2(out_gpu.ctypes.data_as(ctypes.c_void_p), ctypes.c_uint64(self._d_final_out), out_gpu.nbytes)

        # 8. Verificación numérica
        cos_sim = 1.0
        max_err = 0.0
        if verify_against_cpu_ref:
            out_ref = self.compute_cpu_reference(layer_id, expert_id, x_fp32)
            cos_sim = float(np.dot(out_gpu.flatten(), out_ref.flatten()) / (np.linalg.norm(out_gpu) * np.linalg.norm(out_ref)))
            max_err = float(np.max(np.abs(out_gpu - out_ref)))

        total_expert_vram = (2 * self.hidden_dim * self.in_dim * 4) + (self.in_dim * self.hidden_dim * 4)

        return SingleExpertExecutionResult(
            layer_id=layer_id,
            expert_id=expert_id,
            output=out_gpu,
            compute_time_ms=t_compute_ms,
            transfer_time_ms=t_transfer_ms,
            total_time_ms=t_transfer_ms + t_compute_ms,
            vram_bytes_used=total_expert_vram,
            cosine_similarity_vs_ref=cos_sim,
            max_absolute_error_vs_ref=max_err,
        )

    def release(self) -> None:
        """Libera todos los buffers de VRAM al terminar."""
        if self.driver.is_initialized:
            for ptr in [self._d_x, self._d_gate_out, self._d_up_out, self._d_hidden, self._d_final_out, self._d_w_gate, self._d_w_up, self._d_w_down]:
                if ptr != 0:
                    try:
                        self.driver.mem_free(ptr)
                    except Exception:
                        pass
        self._current_loaded_expert = None
        self.cublas.destroy()
        logger.info("[SingleExpertExecutor] Recursos de GPU liberados.")

    def __del__(self):
        self.release()
