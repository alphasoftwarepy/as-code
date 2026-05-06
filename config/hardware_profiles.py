"""
AS Code — Hardware Profile Definitions

Pre-defined hardware profiles that the system auto-selects based
on detected hardware. Each profile optimizes for different tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from core.hardware import HardwareTier


@dataclass(frozen=True)
class HardwareProfile:
    """Inference parameters for a hardware tier."""
    name: str
    tier: HardwareTier
    context_size: int
    quantization: str
    max_vram_mb: int
    active_model_policy: str  # "single" or "multi"
    swap_strategy: str  # "aggressive", "smart", "none"
    batch_size: int
    concurrent_requests: int
    kv_cache_max_mb: int
    model_unload_timeout_sec: float
    anti_oom_threshold_mb: int
    enable_speculative: bool


# Pre-defined profiles
PROFILES: dict[HardwareTier, HardwareProfile] = {
    HardwareTier.ULTRA_LIGHT: HardwareProfile(
        name="Ultra-Light",
        tier=HardwareTier.ULTRA_LIGHT,
        context_size=512,
        quantization="int4",
        max_vram_mb=1500,
        active_model_policy="single",
        swap_strategy="aggressive",
        batch_size=1,
        concurrent_requests=1,
        kv_cache_max_mb=128,
        model_unload_timeout_sec=120.0,
        anti_oom_threshold_mb=300,
        enable_speculative=False,
    ),
    HardwareTier.BALANCED: HardwareProfile(
        name="Balanced",
        tier=HardwareTier.BALANCED,
        context_size=2048,
        quantization="int4",
        max_vram_mb=3200,
        active_model_policy="single",
        swap_strategy="smart",
        batch_size=1,
        concurrent_requests=1,
        kv_cache_max_mb=512,
        model_unload_timeout_sec=300.0,
        anti_oom_threshold_mb=500,
        enable_speculative=True,
    ),
    HardwareTier.PERFORMANCE: HardwareProfile(
        name="Performance",
        tier=HardwareTier.PERFORMANCE,
        context_size=4096,
        quantization="int8",
        max_vram_mb=7000,
        active_model_policy="multi",
        swap_strategy="none",
        batch_size=4,
        concurrent_requests=4,
        kv_cache_max_mb=2048,
        model_unload_timeout_sec=600.0,
        anti_oom_threshold_mb=1000,
        enable_speculative=True,
    ),
}


def get_profile(tier: HardwareTier) -> HardwareProfile:
    """Get the hardware profile for a given tier."""
    return PROFILES[tier]
