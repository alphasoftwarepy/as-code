"""
AS Code — Inference Provider Layer

Fully abstracted provider interface enabling hot-swap between:
- LiteRT-LM CLI (subprocess)
- CompiledModel API (ai-edge-litert)
- Native LiteRT-LM Python API (future)
- NPU runtimes (future)

No API, router, or UI layer changes required when switching providers.
"""

from providers.base import InferenceProvider, InferenceResult, ProviderCapabilities
from providers.registry import ProviderRegistry

__all__ = [
    "InferenceProvider",
    "InferenceResult",
    "ProviderCapabilities",
    "ProviderRegistry",
]
