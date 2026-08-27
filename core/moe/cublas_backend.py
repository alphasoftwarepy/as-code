"""
AS-Core MoE Engine — cuBLAS Hardware Accelerator
=================================================
Wrapper para operaciones GEMM y álgebra lineal sobre GPU NVIDIA usando cublas64_12.dll.
Proporciona ejecución de alta precisión para proyecciones de expertos individuales.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from core.moe.cuda_driver import CUDADriver, CUDADriverError

logger = logging.getLogger("as-code.core.moe.cublas")


class CuBLASError(RuntimeError):
    """Excepción para errores de la API de cuBLAS."""
    def __init__(self, func_name: str, status: int):
        super().__init__(f"cuBLAS Error en {func_name}: código {status}")
        self.func_name = func_name
        self.status = status


class CuBLASBackend:
    """Gestiona el handle de cuBLAS y la ejecución de GEMMs en GPU."""

    def __init__(self, cuda_driver: Optional[CUDADriver] = None, bins_dir: Optional[str] = None):
        self.driver = cuda_driver or CUDADriver()
        self._lib = None
        self._handle = None
        self.is_available = False
        
        # Localizar cublas64_12.dll
        search_dirs = [
            bins_dir or r"C:\as-code\moe_poc\bins",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
            r"C:\Windows\System32",
        ]
        
        for d in search_dirs:
            p = Path(d) / "cublas64_12.dll"
            if p.exists():
                try:
                    if hasattr(os, "add_dll_directory"):
                        os.add_dll_directory(str(p.parent))
                    self._lib = ctypes.windll.LoadLibrary(str(p))
                    logger.info(f"[CuBLASBackend] DLL cargada desde: {p}")
                    break
                except Exception as e:
                    logger.warning(f"Error cargando {p}: {e}")

        if self._lib is not None and self.driver.is_initialized:
            self._init_handle()

    def _init_handle(self) -> None:
        try:
            self._handle = ctypes.c_void_p()
            res = self._lib.cublasCreate_v2(ctypes.byref(self._handle))
            if res != 0:
                raise CuBLASError("cublasCreate_v2", res)
            self.is_available = True
            logger.info("[CuBLASBackend] Handle cuBLAS inicializado exitosamente.")
        except Exception as e:
            logger.warning(f"[CuBLASBackend] No se pudo inicializar handle cuBLAS: {e}")
            self.is_available = False

    def sgemm(
        self,
        trans_a: bool,
        trans_b: bool,
        m: int,
        n: int,
        k: int,
        alpha: float,
        d_a_ptr: int,
        lda: int,
        d_b_ptr: int,
        ldb: int,
        beta: float,
        d_c_ptr: int,
        ldc: int,
    ) -> None:
        """Ejecuta una multiplicación matriz-matriz SGEMM en precisión Float32:
        C = alpha * op(A) * op(B) + beta * C
        """
        if not self.is_available or self._handle is None:
            raise RuntimeError("cuBLAS no está disponible.")

        op_a = 1 if trans_a else 0 # 1=CUBLAS_OP_T, 0=CUBLAS_OP_N
        op_b = 1 if trans_b else 0
        c_alpha = ctypes.c_float(alpha)
        c_beta = ctypes.c_float(beta)

        res = self._lib.cublasSgemm_v2(
            self._handle,
            op_a,
            op_b,
            m,
            n,
            k,
            ctypes.byref(c_alpha),
            ctypes.c_void_p(d_a_ptr),
            lda,
            ctypes.c_void_p(d_b_ptr),
            ldb,
            ctypes.byref(c_beta),
            ctypes.c_void_p(d_c_ptr),
            ldc,
        )
        if res != 0:
            raise CuBLASError("cublasSgemm_v2", res)

    def linear_forward_row_major(
        self,
        d_x_ptr: int,
        d_w_ptr: int,
        d_out_ptr: int,
        batch_size: int,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        """Calcula Y = X @ W.T para tensores Row-Major (C-Order):
        - X: [batch_size, in_features]
        - W: [out_features, in_features]
        - Y: [batch_size, out_features]
        """
        # Mapeo a cuBLAS column-major:
        # Y_col [out_features, batch_size] = W_col [out_features, in_features] @ X_col [in_features, batch_size]
        # Dado que W en memoria C-order es (out_features, in_features), en col-major es (in_features, out_features) con lda = in_features.
        # Por tanto, aplicamos CUBLAS_OP_T sobre W con lda = in_features para obtener (out_features, in_features).
        self.sgemm(
            trans_a=True,   # OP_T sobre W
            trans_b=False,  # OP_N sobre X
            m=out_features,
            n=batch_size,
            k=in_features,
            alpha=alpha,
            d_a_ptr=d_w_ptr,
            lda=in_features,
            d_b_ptr=d_x_ptr,
            ldb=in_features,
            beta=beta,
            d_c_ptr=d_out_ptr,
            ldc=out_features,
        )

    def saxpy(
        self,
        n: int,
        alpha: float,
        d_x_ptr: int,
        incx: int,
        d_y_ptr: int,
        incy: int,
    ) -> None:
        """Calcula Y = alpha * X + Y directamente en la GPU (Float32)."""
        if not self.is_available or self._handle is None:
            raise RuntimeError("cuBLAS no está disponible.")

        c_alpha = ctypes.c_float(alpha)
        res = self._lib.cublasSaxpy_v2(
            self._handle,
            n,
            ctypes.byref(c_alpha),
            ctypes.c_void_p(d_x_ptr),
            incx,
            ctypes.c_void_p(d_y_ptr),
            incy,
        )
        if res != 0:
            raise CuBLASError("cublasSaxpy_v2", res)

    def destroy(self) -> None:
        if self._handle is not None and self._lib is not None:
            try:
                self._lib.cublasDestroy_v2(self._handle)
            except Exception:
                pass
            self._handle = None
            self.is_available = False

    def __del__(self):
        self.destroy()
