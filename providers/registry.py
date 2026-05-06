"""
AS Code — Provider Registry & Hot-Swap Manager

Manages registered providers and enables runtime hot-swapping
between backends without touching API, router, or UI layers.

Usage:
    registry = ProviderRegistry()
    registry.register("litert_cli", LiteRTCLIProvider(...))
    registry.set_active("litert_cli")

    # Hot-swap at runtime:
    registry.register("litert_native", LiteRTNativeProvider(...))
    await registry.hot_swap("litert_native")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from providers.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
    ProviderStatus,
    ProviderType,
)

logger = logging.getLogger("as-code.providers.registry")


class ProviderRegistry:
    """Central registry for inference providers.

    Responsibilities:
    - Register/unregister providers
    - Manage active provider selection
    - Enable zero-downtime hot-swap between providers
    - Provide a stable interface for the engine layer
    """

    def __init__(self) -> None:
        self._providers: dict[str, InferenceProvider] = {}
        self._active_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._swap_in_progress = False

    # ── Registration ───────────────────────────────────────────

    def register(self, provider_id: str, provider: InferenceProvider) -> None:
        """Register a provider. Does NOT initialize or activate it."""
        if provider_id in self._providers:
            logger.warning(f"Overwriting existing provider: {provider_id}")
        self._providers[provider_id] = provider
        logger.info(
            f"Registered provider: {provider_id} "
            f"(type={provider.capabilities().provider_type.value})"
        )

    def unregister(self, provider_id: str) -> None:
        """Remove a provider from the registry.
        Cannot unregister the active provider."""
        if provider_id == self._active_id:
            raise RuntimeError(
                f"Cannot unregister active provider '{provider_id}'. "
                "Switch to another provider first."
            )
        if provider_id in self._providers:
            del self._providers[provider_id]
            logger.info(f"Unregistered provider: {provider_id}")

    # ── Active Provider ────────────────────────────────────────

    @property
    def active_provider(self) -> Optional[InferenceProvider]:
        """Get the currently active provider instance."""
        if self._active_id is None:
            return None
        return self._providers.get(self._active_id)

    @property
    def active_provider_id(self) -> Optional[str]:
        return self._active_id

    @property
    def is_ready(self) -> bool:
        """True if an active provider exists and is ready."""
        p = self.active_provider
        return p is not None and p.status == ProviderStatus.READY

    async def set_active(self, provider_id: str) -> None:
        """Set the active provider. Initializes it if needed."""
        if provider_id not in self._providers:
            raise KeyError(f"Provider '{provider_id}' not registered")

        provider = self._providers[provider_id]

        if provider.status == ProviderStatus.UNINITIALIZED:
            logger.info(f"Initializing provider: {provider_id}")
            await provider.initialize()

        self._active_id = provider_id
        logger.info(f"Active provider set to: {provider_id}")

    # ── Hot-Swap ───────────────────────────────────────────────

    async def hot_swap(self, new_provider_id: str) -> None:
        """Hot-swap to a different provider with zero downtime.

        Flow:
        1. Initialize new provider
        2. Verify it's healthy
        3. Switch active pointer (atomic)
        4. Shutdown old provider in background

        If new provider fails to initialize, the old one stays active.
        """
        if new_provider_id not in self._providers:
            raise KeyError(f"Provider '{new_provider_id}' not registered")

        if new_provider_id == self._active_id:
            logger.info(f"Provider '{new_provider_id}' is already active")
            return

        async with self._lock:
            if self._swap_in_progress:
                raise RuntimeError("Another hot-swap is already in progress")
            self._swap_in_progress = True

        old_id = self._active_id
        old_provider = self.active_provider

        try:
            new_provider = self._providers[new_provider_id]

            # Step 1: Initialize new provider
            if new_provider.status == ProviderStatus.UNINITIALIZED:
                logger.info(f"Hot-swap: initializing {new_provider_id}")
                await new_provider.initialize()

            # Step 2: Health check
            if not await new_provider.health_check():
                raise RuntimeError(
                    f"Provider '{new_provider_id}' failed health check"
                )

            # Step 3: Atomic switch
            self._active_id = new_provider_id
            logger.info(
                f"Hot-swap complete: {old_id} → {new_provider_id}"
            )

            # Step 4: Shutdown old provider in background
            if old_provider is not None:
                asyncio.create_task(
                    self._background_shutdown(old_id, old_provider)
                )

        except Exception as e:
            logger.error(f"Hot-swap to '{new_provider_id}' failed: {e}")
            # Rollback: keep old provider active
            self._active_id = old_id
            raise
        finally:
            self._swap_in_progress = False

    async def _background_shutdown(
        self, provider_id: str, provider: InferenceProvider
    ) -> None:
        """Shutdown a provider in the background after hot-swap."""
        try:
            logger.info(f"Background shutdown: {provider_id}")
            await provider.shutdown()
            logger.info(f"Background shutdown complete: {provider_id}")
        except Exception as e:
            logger.error(f"Background shutdown failed for {provider_id}: {e}")

    # ── Query ──────────────────────────────────────────────────

    def list_providers(self) -> dict[str, dict]:
        """List all registered providers with their status."""
        result = {}
        for pid, provider in self._providers.items():
            caps = provider.capabilities()
            result[pid] = {
                "type": caps.provider_type.value,
                "status": provider.status.value,
                "is_active": pid == self._active_id,
                "supports_gpu": caps.supports_gpu,
                "supports_streaming": caps.supports_streaming,
            }
        return result

    def get_provider(self, provider_id: str) -> Optional[InferenceProvider]:
        """Get a specific provider by ID."""
        return self._providers.get(provider_id)

    # ── Lifecycle ──────────────────────────────────────────────

    async def shutdown_all(self) -> None:
        """Shutdown all providers. Called on application exit."""
        logger.info("Shutting down all providers...")
        for pid, provider in self._providers.items():
            if provider.status not in (
                ProviderStatus.SHUTDOWN,
                ProviderStatus.UNINITIALIZED,
            ):
                try:
                    await provider.shutdown()
                    logger.info(f"Shutdown: {pid}")
                except Exception as e:
                    logger.error(f"Shutdown failed for {pid}: {e}")
        self._active_id = None
        logger.info("All providers shut down")
