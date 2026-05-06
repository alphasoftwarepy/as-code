"""
AS Code — Abstract Inference Provider Interface

This is the core abstraction that ALL inference backends must implement.
The system can hot-swap between providers at runtime without touching
the API, router, or UI layers.

Supported provider types:
- LiteRT-LM CLI (subprocess-based, Windows GPU native)
- CompiledModel API (ai-edge-litert, DirectX GPU)
- Native LiteRT-LM Python API (future, when Windows support lands)
- NPU runtimes (future hardware acceleration)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional


class ProviderType(str, Enum):
    """Enumeration of supported provider backends."""
    LITERT_CLI = "litert_cli"
    LITERT_COMPILED = "litert_compiled"
    LITERT_NATIVE = "litert_native"
    NPU = "npu"


class ProviderStatus(str, Enum):
    """Current status of a provider instance."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class ProviderCapabilities:
    """Declares what a provider can do. Used by the engine to make
    decisions without coupling to provider internals."""
    supports_gpu: bool = False
    supports_npu: bool = False
    supports_streaming: bool = False
    supports_speculative_decoding: bool = False
    supports_multi_model: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    max_context_length: int = 2048
    supported_quantizations: tuple[str, ...] = ("int4",)
    provider_type: ProviderType = ProviderType.LITERT_CLI


@dataclass
class InferenceResult:
    """Single inference result chunk (for both streaming and non-streaming)."""
    text: str = ""
    finish_reason: Optional[str] = None  # "stop", "length", "error"
    tokens_generated: int = 0
    prompt_tokens: int = 0
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    model_id: str = ""
    provider_type: str = ""


@dataclass
class InferenceRequest:
    """Standardized inference request passed to any provider.
    Decouples the OpenAI API format from provider internals."""
    prompt: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    top_k: int = 40
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = True
    system_prompt: Optional[str] = None

    # Internal tracking
    request_id: str = ""
    created_at: float = field(default_factory=time.time)


class InferenceProvider(ABC):
    """Abstract base class for all inference providers.

    Every provider MUST implement these methods. The engine manager
    interacts ONLY through this interface, enabling hot-swap between
    CLI, CompiledModel, native Python API, or future NPU backends
    without any changes to API/router/UI layers.

    Lifecycle:
        1. __init__() — lightweight construction, no heavy resources
        2. initialize() — load models, allocate GPU, etc.
        3. generate() / generate_stream() — inference
        4. shutdown() — release all resources

    Thread safety: Providers should be safe for concurrent async calls.
    """

    def __init__(self) -> None:
        self._status: ProviderStatus = ProviderStatus.UNINITIALIZED
        self._last_error: Optional[str] = None

    @property
    def status(self) -> ProviderStatus:
        return self._status

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    # ── Lifecycle ──────────────────────────────────────────────

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider: validate runtime, check GPU, etc.
        Called once before any inference. Must be idempotent."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release all resources: unload models, free VRAM, close processes.
        Must be safe to call multiple times."""
        ...

    # ── Capabilities ───────────────────────────────────────────

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the capabilities of this provider.
        Called by the engine to make runtime decisions."""
        ...

    # ── Model Management ───────────────────────────────────────

    @abstractmethod
    async def load_model(self, model_id: str, model_path: str) -> None:
        """Load a model into memory/VRAM.
        Must handle VRAM-aware loading and fail gracefully on OOM."""
        ...

    @abstractmethod
    async def unload_model(self, model_id: str) -> None:
        """Unload a model, freeing all associated resources."""
        ...

    @abstractmethod
    async def is_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is currently loaded and ready."""
        ...

    @abstractmethod
    async def loaded_models(self) -> list[str]:
        """Return list of currently loaded model IDs."""
        ...

    # ── Inference ──────────────────────────────────────────────

    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Run inference and return the complete result.
        For non-streaming use cases."""
        ...

    @abstractmethod
    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[InferenceResult]:
        """Run inference and yield results token-by-token.
        Primary method for SSE streaming responses."""
        ...

    @abstractmethod
    async def cancel_generation(self, request_id: str) -> None:
        """Cancel an in-progress generation. Must be safe to call
        even if no generation is running for this request_id."""
        ...

    # ── Health & Telemetry ─────────────────────────────────────

    @abstractmethod
    async def health_check(self) -> bool:
        """Quick health check. Return True if provider is operational."""
        ...

    @abstractmethod
    async def get_metrics(self) -> dict:
        """Return provider-specific metrics for telemetry.
        Expected keys: vram_used_mb, ram_used_mb, active_model, etc."""
        ...

    # ── Convenience ────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"status={self._status.value} "
            f"type={self.capabilities().provider_type.value}>"
        )
