"""
AS-Core MoE Engine — CUDA Driver Wrapper
=========================================
Encapsula la interacción directa de bajo nivel con el Driver de CUDA (nvcuda.dll)
para asignaciones de VRAM, transferencias DMA y gestión de memoria GPU en Windows.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from typing import Optional, Tuple, Union

logger = logging.getLogger("as-code.core.moe.cuda")

# Tipos nativos CUDA
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
CUstream = ctypes.c_void_p
CUresult = ctypes.c_int


class CUDADriverError(RuntimeError):
    """Excepción para errores de la API del driver CUDA."""
    def __init__(self, func_name: str, code: int):
        super().__init__(f"CUDA Driver Error en {func_name}: código {code}")
        self.func_name = func_name
        self.code = code


class CUDADriver:
    """Wrapper singleton / gestionado para interactuar con nvcuda.dll."""

    _instance: Optional[CUDADriver] = None

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._lib = None
        self._ctx = None
        self._dev = None
        self.is_initialized = False
        self._init_driver()

    def _init_driver(self) -> None:
        try:
            if platform.system() == "Windows":
                self._lib = ctypes.windll.LoadLibrary("nvcuda.dll")
            else:
                self._lib = ctypes.CDLL("libcuda.so")

            # 1. cuInit(0)
            res = self._lib.cuInit(0)
            if res != 0:
                logger.warning(f"cuInit falló con código {res}. CUDA no disponible.")
                return

            # 2. cuDeviceGet(&dev, device_index)
            self._dev = CUdevice()
            res = self._lib.cuDeviceGet(ctypes.byref(self._dev), self.device_index)
            if res != 0:
                logger.warning(f"cuDeviceGet falló con código {res}.")
                return

            # 3. cuCtxCreate_v2(&ctx, 0, dev)
            self._ctx = CUcontext()
            res = self._lib.cuCtxCreate_v2(ctypes.byref(self._ctx), 0, self._dev)
            if res != 0:
                logger.warning(f"cuCtxCreate falló con código {res}.")
                return

            self.is_initialized = True
            logger.info(f"[CUDADriver] Inicializado exitosamente en GPU Device {self.device_index}.")

        except Exception as e:
            logger.warning(f"[CUDADriver] No se pudo inicializar el driver CUDA: {e}")
            self.is_initialized = False

    def check(self, res: int, func_name: str) -> None:
        if res != 0:
            raise CUDADriverError(func_name, res)

    def get_mem_info(self) -> Tuple[int, int]:
        """Retorna (free_bytes, total_bytes) de la VRAM."""
        if not self.is_initialized:
            return 0, 0
        free_b = ctypes.c_size_t()
        total_b = ctypes.c_size_t()
        res = self._lib.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b))
        self.check(res, "cuMemGetInfo_v2")
        return free_b.value, total_b.value

    def mem_alloc(self, size_bytes: int) -> int:
        """Asigna un buffer contiguo en la VRAM de la GPU y retorna su puntero de 64-bits."""
        if not self.is_initialized:
            raise RuntimeError("CUDA Driver no inicializado.")
        d_ptr = CUdeviceptr()
        res = self._lib.cuMemAlloc_v2(ctypes.byref(d_ptr), ctypes.c_size_t(size_bytes))
        self.check(res, "cuMemAlloc_v2")
        return d_ptr.value

    def mem_free(self, d_ptr: int) -> None:
        """Libera un buffer de VRAM previamente asignado."""
        if not self.is_initialized or d_ptr == 0:
            return
        res = self._lib.cuMemFree_v2(CUdeviceptr(d_ptr))
        self.check(res, "cuMemFree_v2")

    def mem_alloc_host(self, size_bytes: int) -> int:
        """Asigna memoria fijada (pinned / page-locked) en Host RAM para DMA."""
        if not self.is_initialized:
            raise RuntimeError("CUDA Driver no inicializado.")
        h_ptr = ctypes.c_void_p()
        res = self._lib.cuMemAllocHost_v2(ctypes.byref(h_ptr), ctypes.c_size_t(size_bytes))
        self.check(res, "cuMemAllocHost_v2")
        return h_ptr.value

    def mem_free_host(self, h_ptr: int) -> None:
        """Libera memoria fijada de Host."""
        if not self.is_initialized or h_ptr == 0:
            return
        res = self._lib.cuMemFreeHost(ctypes.c_void_p(h_ptr))
        self.check(res, "cuMemFreeHost")

    def memcpy_htod(self, d_dst_ptr: int, h_src_buffer: Union[bytes, bytearray, memoryview, ctypes.Array, int], size_bytes: int) -> None:
        """Copia síncrona de Host a Device (VRAM) con soporte multitipe (bytes, bytearray, pointers)."""
        if not self.is_initialized:
            raise RuntimeError("CUDA Driver no inicializado.")
        
        if isinstance(h_src_buffer, int):
            c_src = ctypes.c_void_p(h_src_buffer)
        elif isinstance(h_src_buffer, bytearray):
            c_src = (ctypes.c_char * size_bytes).from_buffer(h_src_buffer)
        elif isinstance(h_src_buffer, (bytes, memoryview)):
            c_src = ctypes.c_char_p(bytes(h_src_buffer))
        else:
            c_src = h_src_buffer

        res = self._lib.cuMemcpyHtoD_v2(CUdeviceptr(d_dst_ptr), c_src, ctypes.c_size_t(size_bytes))
        self.check(res, "cuMemcpyHtoD_v2")

    def memcpy_htod_async(self, d_dst_ptr: int, h_src_ptr: int, size_bytes: int, stream: Optional[int] = None) -> None:
        """Copia asíncrona de Host (Pinned) a Device (VRAM) usando un CUDA stream."""
        if not self.is_initialized:
            raise RuntimeError("CUDA Driver no inicializado.")
        st = CUstream(stream) if stream else CUstream(0)
        res = self._lib.cuMemcpyHtoDAsync_v2(CUdeviceptr(d_dst_ptr), ctypes.c_void_p(h_src_ptr), ctypes.c_size_t(size_bytes), st)
        self.check(res, "cuMemcpyHtoDAsync_v2")

    def synchronize(self) -> None:
        """Sincroniza el contexto CUDA asegurando que todas las operaciones finalizaron."""
        if not self.is_initialized:
            return
        res = self._lib.cuCtxSynchronize()
        self.check(res, "cuCtxSynchronize")

    def destroy(self) -> None:
        """Destruye el contexto CUDA si existe."""
        if self._ctx is not None and self._lib is not None:
            try:
                self._lib.cuCtxDestroy_v2(self._ctx)
            except Exception:
                pass
            self._ctx = None
            self.is_initialized = False
