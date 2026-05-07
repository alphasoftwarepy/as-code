"""
AS Code — Engine Manager

Central orchestrator that sits between the API layer and providers.
The engine manager:
1. Owns the ProviderRegistry
2. Manages model lifecycle (lazy load, swap, unload)
3. Enforces hardware-adaptive policies
4. Delegates inference to the active provider

API/router/UI layers ONLY talk to the EngineManager.
They never touch providers directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

from core.hardware import (
    HardwareInfo,
    HardwareTier,
    detect_hardware,
    get_ram_available_mb,
    get_vram_free_mb,
)
from providers.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderStatus,
)
from providers.registry import ProviderRegistry

logger = logging.getLogger("as-code.core.engine")


class EngineManager:
    """Central engine orchestrator.

    Responsibilities:
    - Hardware-adaptive inference policy
    - Model lifecycle management (lazy loading, timeout-based unloading)
    - VRAM/RAM-aware loading with anti-OOM protection
    - Provider delegation (all inference goes through active provider)

    The engine is the ONLY component that interacts with providers.
    """

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        hardware_info: Optional[HardwareInfo] = None,
        max_vram_mb: int = 3200,
        model_unload_timeout: float = 300.0,
        anti_oom_threshold_mb: int = 500,
    ) -> None:
        self.registry = provider_registry
        self.hardware = hardware_info or detect_hardware()

        # Hardware-adaptive settings
        self._max_vram_mb = max_vram_mb
        self._model_unload_timeout = model_unload_timeout
        self._anti_oom_threshold_mb = anti_oom_threshold_mb

        # Model tracking
        self._model_configs: dict[str, dict] = {}  # model_id → config
        self._last_used: dict[str, float] = {}  # model_id → timestamp
        self._active_model: Optional[str] = None

        # Background tasks
        self._unload_task: Optional[asyncio.Task] = None
        self._running = False

        # Apply hardware profile
        self._apply_hardware_profile()

    def _apply_hardware_profile(self) -> None:
        """Apply hardware-adaptive settings based on detected tier."""
        tier = self.hardware.tier

        if tier == HardwareTier.ULTRA_LIGHT:
            self._max_vram_mb = min(self._max_vram_mb, 1500)
            self._model_unload_timeout = 120.0
            self._anti_oom_threshold_mb = 300
            logger.info("Applied ultra-light hardware profile")

        elif tier == HardwareTier.BALANCED:
            self._max_vram_mb = min(
                self._max_vram_mb,
                int(self.hardware.gpu.vram_total_mb * 0.8)
            )
            self._model_unload_timeout = 300.0
            self._anti_oom_threshold_mb = 500
            logger.info("Applied balanced hardware profile")

        elif tier == HardwareTier.PERFORMANCE:
            self._max_vram_mb = int(self.hardware.gpu.vram_total_mb * 0.9)
            self._model_unload_timeout = 600.0
            self._anti_oom_threshold_mb = 1000
            logger.info("Applied performance hardware profile")

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the engine manager and background tasks."""
        self._running = True
        self._unload_task = asyncio.create_task(self._unload_loop())
        logger.info(
            f"Engine started | {self.hardware.summary()} | "
            f"Max VRAM: {self._max_vram_mb}MB"
        )

    async def stop(self) -> None:
        """Stop the engine and shutdown all providers."""
        self._running = False
        if self._unload_task:
            self._unload_task.cancel()
            try:
                await self._unload_task
            except asyncio.CancelledError:
                pass
        await self.registry.shutdown_all()
        logger.info("Engine stopped")

    # ── Model Registration ─────────────────────────────────────

    def register_model(
        self,
        model_id: str,
        model_path: str,
        model_type: str = "general",
        estimated_vram_mb: int = 1500,
        provider_id: Optional[str] = None,
    ) -> None:
        """Register a model configuration.
        Does NOT load the model — loading is lazy."""
        self._model_configs[model_id] = {
            "path": model_path,
            "type": model_type,
            "estimated_vram_mb": estimated_vram_mb,
            "provider_id": provider_id,
        }
        logger.info(f"Model registered: {model_id} (type={model_type})")

    # ── Inference ──────────────────────────────────────────────

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Run inference (non-streaming). Handles model loading."""
        provider = await self._ensure_model_loaded(request.model_id)

        if provider is None:
            return InferenceResult(
                text="No active inference provider",
                finish_reason="error",
                model_id=request.model_id,
            )

        self._last_used[request.model_id] = time.time()
        return await provider.generate(request)

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[InferenceResult]:
        """Run inference (streaming). Handles model loading."""
        provider = await self._ensure_model_loaded(request.model_id)

        if provider is None:
            yield InferenceResult(
                text="No active inference provider",
                finish_reason="error",
                model_id=request.model_id,
            )
            return

        self._last_used[request.model_id] = time.time()
        async for chunk in provider.generate_stream(request):
            yield chunk

    async def cancel_generation(self, request_id: str, model_id: Optional[str] = None) -> None:
        """Cancel an in-progress generation."""
        if model_id and model_id in self._model_configs:
            config = self._model_configs[model_id]
            provider_id = config.get("provider_id")
            provider = self.registry.get_provider(provider_id) if provider_id else self.registry.active_provider
        else:
            provider = self.registry.active_provider
            
        if provider:
            await provider.cancel_generation(request_id)

    # ── Model Loading ──────────────────────────────────────────

    async def _ensure_model_loaded(self, model_id: str) -> InferenceProvider:
        """Ensure a model is loaded. Lazy load if needed.
        Implements anti-OOM protection and VRAM-aware loading."""
        if model_id not in self._model_configs:
            raise ValueError(f"Model '{model_id}' not registered")

        config = self._model_configs[model_id]
        provider_id = config.get("provider_id")
        
        provider = self.registry.get_provider(provider_id) if provider_id else self.registry.active_provider
        
        if provider is None:
            raise RuntimeError(f"No provider found for model '{model_id}' (provider_id={provider_id})")

        if await provider.is_model_loaded(model_id):
            return provider

        # Anti-OOM check
        await self._check_resources(config["estimated_vram_mb"])

        # If another model is loaded and we're in single-model mode,
        # unload it first
        if self.hardware.tier != HardwareTier.PERFORMANCE:
            loaded = await provider.loaded_models()
            for loaded_id in loaded:
                if loaded_id != model_id:
                    logger.info(f"Swapping model: {loaded_id} → {model_id}")
                    await provider.unload_model(loaded_id)

        # Load the model
        logger.info(f"Loading model: {model_id}")
        print("ENGINE CONFIG PATH:", config["path"])
        await provider.load_model(model_id, config["path"])
        self._active_model = model_id
        self._last_used[model_id] = time.time()
        return provider

    async def _check_resources(self, required_vram_mb: int) -> None:
        """Check if we have enough resources to load a model.
        Anti-OOM protection with real-world latency priority."""
        # RAM check
        available_ram = get_ram_available_mb()
        if available_ram > 0 and available_ram < self._anti_oom_threshold_mb:
            logger.warning(
                f"Low RAM: {available_ram}MB available "
                f"(threshold: {self._anti_oom_threshold_mb}MB)"
            )
            # Don't block — let the OS manage swap
            # but log aggressively so we can tune

        # VRAM check
        free_vram = get_vram_free_mb()
        if free_vram > 0 and required_vram_mb > free_vram:
            logger.warning(
                f"VRAM may be insufficient: need ~{required_vram_mb}MB, "
                f"have {free_vram}MB free"
            )

    # ── Background Tasks ───────────────────────────────────────

    async def _unload_loop(self) -> None:
        """Background loop to unload idle models (dynamic unloading)."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                provider = self.registry.active_provider
                if provider is None:
                    continue

                now = time.time()
                loaded = await provider.loaded_models()

                for model_id in loaded:
                    last_used = self._last_used.get(model_id, 0)
                    idle_time = now - last_used

                    if idle_time > self._model_unload_timeout:
                        logger.info(
                            f"Unloading idle model: {model_id} "
                            f"(idle for {idle_time:.0f}s)"
                        )
                        await provider.unload_model(model_id)
                        self._last_used.pop(model_id, None)
                        if self._active_model == model_id:
                            self._active_model = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unload loop error: {e}")

    # ── Status & Metrics ───────────────────────────────────────

    @property
    def active_model(self) -> Optional[str]:
        return self._active_model

    async def get_status(self) -> dict:
        """Get engine status for /v1/status endpoint."""
        provider = self.registry.active_provider
        provider_metrics = {}
        if provider:
            provider_metrics = await provider.get_metrics()

        return {
            "active_model": self._active_model,
            "hardware_tier": self.hardware.tier.value,
            "max_vram_mb": self._max_vram_mb,
            "gpu": {
                "name": self.hardware.gpu.name,
                "vram_total_mb": self.hardware.gpu.vram_total_mb,
                "vram_free_mb": get_vram_free_mb() or self.hardware.gpu.vram_free_mb,
            },
            "ram_available_mb": get_ram_available_mb() or self.hardware.memory.available_mb,
            "provider": provider_metrics,
            "registered_models": list(self._model_configs.keys()),
        }

    def get_registered_models(self) -> list[dict]:
        """Get list of registered models for /v1/models endpoint."""
        models = []
        for model_id, config in self._model_configs.items():
            models.append({
                "id": model_id,
                "object": "model",
                "owned_by": "as-code",
                "type": config["type"],
                "estimated_vram_mb": config["estimated_vram_mb"],
            })
        return models
