"""
AS Code — Inference Profiles

Per-model inference configurations optimized for real-world latency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceProfile:
    """Model-specific inference parameters."""
    model_id: str
    temperature: float
    max_tokens: int
    top_k: int
    top_p: float
    context_length: int
    system_prompt: str
    quantization: str
    estimated_vram_mb: int
    supports_speculative: bool


# Default profiles for supported models
INFERENCE_PROFILES: dict[str, InferenceProfile] = {
    "chat": InferenceProfile(
        model_id="chat",
        temperature=0.7,
        max_tokens=5120,
        top_k=40,
        top_p=0.95,
        context_length=2048,
        system_prompt=(
            "You are a helpful, friendly, general-purpose AI assistant. "
            "You can engage in natural conversation, planning, brainstorming, "
            "explanations, and analysis. Respond naturally and clearly. "
            "Only generate code when explicitly requested."
        ),
        quantization="int4",
        estimated_vram_mb=1500,
        supports_speculative=True,
    ),
    "code": InferenceProfile(
        model_id="code",
        temperature=0.7,
        max_tokens=5120,
        top_k=40,
        top_p=0.95,
        context_length=2048,
        system_prompt=(
            "You are an expert software engineering assistant specialized in "
            "programming, debugging, APIs, architecture, and development workflows. "
            "Write clean, efficient, production-ready code."
        ),
        quantization="int4",
        estimated_vram_mb=1500,
        supports_speculative=True,
    ),
}


def get_inference_profile(model_id: str) -> InferenceProfile:
    """Get inference profile for a model. Falls back to defaults."""
    if model_id in INFERENCE_PROFILES:
        return INFERENCE_PROFILES[model_id]

    # Default fallback profile
    return InferenceProfile(
        model_id=model_id,
        temperature=0.7,
        max_tokens=5120,
        top_k=40,
        top_p=0.95,
        context_length=2048,
        system_prompt="You are a helpful AI assistant.",
        quantization="int4",
        estimated_vram_mb=1500,
        supports_speculative=False,
    )
