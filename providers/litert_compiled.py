"""
AS Code — LiteRT CompiledModel API Provider (Stub)

Secondary provider using the ai-edge-litert CompiledModel API
with DirectX GPU acceleration on Windows.

STATUS: Stub implementation. The CompiledModel API provides lower-level
control over inference but requires manual tokenization, KV-cache management,
and output decoding. This provider will be fully implemented when the
ai-edge-litert package matures for LLM workloads on Windows.

The provider interface is fully implemented so it can be registered
in the ProviderRegistry and hot-swapped to at any time.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from providers.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
    ProviderStatus,
    ProviderType,
)

logger = logging.getLogger("as-code.providers.litert_compiled")


class LiteRTCompiledProvider(InferenceProvider):
    """Inference provider using ai-edge-litert CompiledModel API.

    Requires:
    - pip install ai-edge-litert
    - DirectXShaderCompiler DLLs (dxil.dll, dxcompiler.dll)

    This provider offers fine-grained control over inference but
    requires manual management of tokenization and decoding.
    """

    def __init__(self, models_dir: str = "models") -> None:
        super().__init__()
        self._models_dir = models_dir
        self._compiled_models: dict[str, object] = {}
        self._available = False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_gpu=True,
            supports_npu=False,
            supports_streaming=False,  # requires manual implementation
            supports_speculative_decoding=False,
            supports_multi_model=True,
            supports_vision=False,
            supports_audio=False,
            max_context_length=2048,
            supported_quantizations=("int4", "int8", "fp16"),
            provider_type=ProviderType.LITERT_COMPILED,
        )

    async def initialize(self) -> None:
        """Check if ai-edge-litert is installed."""
        self._status = ProviderStatus.INITIALIZING
        try:
            import ai_edge_litert  # noqa: F401
            self._available = True
            self._status = ProviderStatus.READY
            logger.info("CompiledModel API available")
        except ImportError:
            self._available = False
            self._status = ProviderStatus.READY
            self._last_error = (
                "ai-edge-litert not installed. "
                "Install with: pip install ai-edge-litert"
            )
            logger.warning(self._last_error)

    async def shutdown(self) -> None:
        self._compiled_models.clear()
        self._status = ProviderStatus.SHUTDOWN

    async def load_model(self, model_id: str, model_path: str) -> None:
        if not self._available:
            raise RuntimeError("ai-edge-litert not installed")
        # Stub: actual CompiledModel loading would go here
        logger.info(f"[STUB] Would load CompiledModel: {model_id}")
        self._compiled_models[model_id] = model_path

    async def unload_model(self, model_id: str) -> None:
        self._compiled_models.pop(model_id, None)

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._compiled_models

    async def loaded_models(self) -> list[str]:
        return list(self._compiled_models.keys())

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            text="[CompiledModel provider not yet implemented]",
            finish_reason="error",
            model_id=request.model_id,
            provider_type=ProviderType.LITERT_COMPILED.value,
        )

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[InferenceResult]:
        yield InferenceResult(
            text="[CompiledModel provider not yet implemented]",
            finish_reason="error",
            model_id=request.model_id,
            provider_type=ProviderType.LITERT_COMPILED.value,
        )

    async def cancel_generation(self, request_id: str) -> None:
        pass  # No active processes to cancel

    async def health_check(self) -> bool:
        return self._status == ProviderStatus.READY

    async def get_metrics(self) -> dict:
        return {
            "provider_type": ProviderType.LITERT_COMPILED.value,
            "status": self._status.value,
            "available": self._available,
            "loaded_models": list(self._compiled_models.keys()),
        }
