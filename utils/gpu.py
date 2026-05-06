"""
AS Code — GPU Detection Utilities

Lightweight GPU detection without heavy dependencies.
Optimized for NVIDIA GPUs on Windows.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

logger = logging.getLogger("as-code.utils.gpu")


def get_nvidia_gpu_info() -> Optional[dict]:
    """Get NVIDIA GPU info via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,"
                "temperature.gpu,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 7:
                return {
                    "name": parts[0],
                    "vram_total_mb": int(float(parts[1])),
                    "vram_free_mb": int(float(parts[2])),
                    "vram_used_mb": int(float(parts[3])),
                    "temperature_c": int(float(parts[4])),
                    "utilization_pct": int(float(parts[5])),
                    "driver_version": parts[6],
                }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        logger.debug(f"nvidia-smi not available: {e}")
    return None


def is_gpu_available() -> bool:
    """Quick check if any GPU is available."""
    info = get_nvidia_gpu_info()
    return info is not None


def get_vram_usage() -> dict:
    """Get current VRAM usage."""
    info = get_nvidia_gpu_info()
    if info:
        return {
            "total_mb": info["vram_total_mb"],
            "used_mb": info["vram_used_mb"],
            "free_mb": info["vram_free_mb"],
            "utilization_pct": info.get("utilization_pct", 0),
        }
    return {"total_mb": 0, "used_mb": 0, "free_mb": 0, "utilization_pct": 0}
