"""
AS Code — Memory Management Helpers

Anti-OOM protection and memory-aware decision making.
"""

from __future__ import annotations

import gc
import logging
import platform
import subprocess
from typing import Optional

logger = logging.getLogger("as-code.utils.memory")


def force_gc() -> None:
    """Force garbage collection to free Python memory."""
    gc.collect()
    gc.collect()  # Second pass for cyclic references


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def is_memory_pressure() -> bool:
    """Check if system is under memory pressure.

    Returns True if available RAM is below 1GB — a signal
    to be conservative with model loading.
    """
    try:
        import psutil
        available = psutil.virtual_memory().available
        return available < 1024 * 1024 * 1024  # < 1GB
    except ImportError:
        pass

    # Fallback for Windows without psutil
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "OS", "get", "FreePhysicalMemory", "/format:list"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip().startswith("FreePhysicalMemory="):
                    free_kb = int(line.strip().split("=")[1])
                    return free_kb < 1024 * 1024  # < 1GB
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

    return False


def estimate_model_memory_mb(
    params_billions: float,
    quantization: str = "int4",
) -> int:
    """Estimate memory needed for a model based on params and quantization.

    Rough estimates:
    - FP16: ~2 bytes/param
    - INT8: ~1 byte/param
    - INT4: ~0.5 bytes/param
    Plus ~20% overhead for KV cache and runtime.
    """
    bytes_per_param = {
        "fp16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }
    bpp = bytes_per_param.get(quantization, 0.5)
    raw_mb = (params_billions * 1e9 * bpp) / (1024 * 1024)
    # Add 20% overhead
    return int(raw_mb * 1.2)
