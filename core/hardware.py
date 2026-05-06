"""
AS Code — Hardware Detection & Profiling

Detects CPU, GPU, RAM, and disk capabilities to auto-select the
optimal inference profile (ultra-light / balanced / performance).

Optimized for real-world latency decisions, NOT synthetic benchmarks.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("as-code.core.hardware")


class HardwareTier(str, Enum):
    """Classified hardware capability tier."""
    ULTRA_LIGHT = "ultra_light"   # ≤8GB RAM, no GPU or ≤2GB VRAM
    BALANCED = "balanced"         # 16GB RAM, 4GB VRAM (target: ASUS TUF)
    PERFORMANCE = "performance"   # ≥32GB RAM, ≥8GB VRAM


@dataclass
class GPUInfo:
    """Detected GPU information."""
    name: str = "unknown"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    driver_version: str = "unknown"
    compute_capability: str = "unknown"
    is_available: bool = False


@dataclass
class CPUInfo:
    """Detected CPU information."""
    name: str = "unknown"
    cores_physical: int = 1
    cores_logical: int = 1
    frequency_mhz: float = 0.0
    has_avx: bool = False
    has_avx2: bool = False


@dataclass
class MemoryInfo:
    """Detected memory information."""
    total_mb: int = 0
    available_mb: int = 0
    used_percent: float = 0.0


@dataclass
class DiskInfo:
    """Detected disk information."""
    total_gb: float = 0.0
    free_gb: float = 0.0
    is_ssd: bool = False


@dataclass
class HardwareInfo:
    """Complete hardware profile."""
    cpu: CPUInfo = field(default_factory=CPUInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    disk: DiskInfo = field(default_factory=DiskInfo)
    tier: HardwareTier = HardwareTier.ULTRA_LIGHT
    os_name: str = "unknown"
    os_version: str = "unknown"

    def summary(self) -> str:
        return (
            f"Tier: {self.tier.value} | "
            f"CPU: {self.cpu.cores_physical}C/{self.cpu.cores_logical}T | "
            f"RAM: {self.memory.total_mb}MB ({self.memory.available_mb}MB free) | "
            f"GPU: {self.gpu.name} ({self.gpu.vram_total_mb}MB VRAM)"
        )


def detect_hardware() -> HardwareInfo:
    """Detect system hardware capabilities.

    Uses lightweight system calls — no heavy dependencies.
    Falls back gracefully if detection fails.
    """
    info = HardwareInfo()
    info.os_name = platform.system()
    info.os_version = platform.version()

    # CPU detection
    info.cpu = _detect_cpu()

    # Memory detection
    info.memory = _detect_memory()

    # GPU detection
    info.gpu = _detect_gpu()

    # Disk detection
    info.disk = _detect_disk()

    # Classify tier
    info.tier = _classify_tier(info)

    logger.info(f"Hardware detected: {info.summary()}")
    return info


def _detect_cpu() -> CPUInfo:
    """Detect CPU information."""
    cpu = CPUInfo()

    try:
        cpu.cores_logical = os.cpu_count() or 1
        cpu.cores_physical = cpu.cores_logical // 2 or 1

        # Get CPU name on Windows
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "Name", "/format:list"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Name="):
                        cpu.name = line.split("=", 1)[1].strip()
                        break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                cpu.name = platform.processor() or "unknown"

            # Check AVX support
            try:
                result = subprocess.run(
                    [
                        "powershell", "-Command",
                        "[System.Runtime.Intrinsics.X86.Avx.IsSupported]"
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                cpu.has_avx = "True" in result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        else:
            cpu.name = platform.processor() or "unknown"

    except Exception as e:
        logger.warning(f"CPU detection failed: {e}")

    return cpu


def _detect_memory() -> MemoryInfo:
    """Detect system memory."""
    mem = MemoryInfo()

    try:
        # Try psutil first (optional dependency)
        try:
            import psutil
            vm = psutil.virtual_memory()
            mem.total_mb = int(vm.total / (1024 * 1024))
            mem.available_mb = int(vm.available / (1024 * 1024))
            mem.used_percent = vm.percent
            return mem
        except ImportError:
            pass

        # Fallback: Windows wmic
        if platform.system() == "Windows":
            result = subprocess.run(
                [
                    "wmic", "OS", "get",
                    "TotalVisibleMemorySize,FreePhysicalMemory",
                    "/format:list",
                ],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("TotalVisibleMemorySize="):
                    mem.total_mb = int(line.split("=")[1]) // 1024
                elif line.startswith("FreePhysicalMemory="):
                    mem.available_mb = int(line.split("=")[1]) // 1024

            if mem.total_mb > 0:
                mem.used_percent = round(
                    (1 - mem.available_mb / mem.total_mb) * 100, 1
                )

    except Exception as e:
        logger.warning(f"Memory detection failed: {e}")

    return mem


def _detect_gpu() -> GPUInfo:
    """Detect GPU using nvidia-smi (NVIDIA) or fallback methods."""
    gpu = GPUInfo()

    try:
        # Try nvidia-smi first
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                gpu.name = parts[0].strip()
                gpu.vram_total_mb = int(float(parts[1].strip()))
                gpu.vram_free_mb = int(float(parts[2].strip()))
                gpu.driver_version = parts[3].strip()
                gpu.is_available = True
                return gpu

    except FileNotFoundError:
        logger.debug("nvidia-smi not found, trying alternative detection")
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"nvidia-smi failed: {e}")

    # Fallback: Windows WMI
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "wmic", "path", "win32_VideoController", "get",
                    "Name,AdapterRAM", "/format:list",
                ],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("Name="):
                    gpu.name = line.split("=", 1)[1].strip()
                elif line.startswith("AdapterRAM="):
                    try:
                        ram_bytes = int(line.split("=")[1])
                        gpu.vram_total_mb = ram_bytes // (1024 * 1024)
                        gpu.vram_free_mb = gpu.vram_total_mb  # estimate
                        gpu.is_available = gpu.vram_total_mb > 512
                    except ValueError:
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return gpu


def _detect_disk() -> DiskInfo:
    """Detect disk space for the current drive."""
    disk = DiskInfo()

    try:
        # Try psutil first
        try:
            import psutil
            usage = psutil.disk_usage(os.getcwd())
            disk.total_gb = round(usage.total / (1024**3), 1)
            disk.free_gb = round(usage.free / (1024**3), 1)
            disk.is_ssd = True  # assume SSD for modern systems
            return disk
        except ImportError:
            pass

        # Fallback: os.statvfs or shutil
        import shutil
        total, used, free = shutil.disk_usage(os.getcwd())
        disk.total_gb = round(total / (1024**3), 1)
        disk.free_gb = round(free / (1024**3), 1)
        disk.is_ssd = True  # conservative assumption

    except Exception as e:
        logger.warning(f"Disk detection failed: {e}")

    return disk


def _classify_tier(info: HardwareInfo) -> HardwareTier:
    """Classify hardware into performance tier.

    Decision priorities (real-world latency first):
    1. VRAM is the primary bottleneck for LLM inference
    2. RAM determines model loading and swap capability
    3. CPU matters for tokenization, not inference
    """
    vram = info.gpu.vram_total_mb
    ram = info.memory.total_mb

    # Performance tier: ≥8GB VRAM and ≥32GB RAM
    if vram >= 8192 and ram >= 32768:
        return HardwareTier.PERFORMANCE

    # Balanced tier: ≥3GB VRAM and ≥12GB RAM
    if vram >= 3072 and ram >= 12288:
        return HardwareTier.BALANCED

    # Everything else is ultra-light
    return HardwareTier.ULTRA_LIGHT


def get_vram_free_mb() -> int:
    """Quick VRAM check for runtime decisions. Returns 0 if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(float(result.stdout.strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def get_ram_available_mb() -> int:
    """Quick RAM check for runtime decisions."""
    try:
        import psutil
        return int(psutil.virtual_memory().available / (1024 * 1024))
    except ImportError:
        pass

    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "OS", "get", "FreePhysicalMemory", "/format:list"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip().startswith("FreePhysicalMemory="):
                    return int(line.strip().split("=")[1]) // 1024
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return 0
