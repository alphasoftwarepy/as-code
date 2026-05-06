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
    "deepseek-r1-1.5b": InferenceProfile(
        model_id="deepseek-r1-1.5b",
        temperature=0.6,
        max_tokens=2048,
        top_k=40,
        top_p=0.95,
        context_length=2048,
        system_prompt=(
            "You are an expert AI assistant specialized in analysis, "
            "planning, and reasoning. Think step-by-step."
        ),
        quantization="int4",
        estimated_vram_mb=1200,
        supports_speculative=False,
    ),
    "gemma-4-e2b": InferenceProfile(
        model_id="gemma-4-e2b",
        temperature=0.7,
        max_tokens=1024,
        top_k=40,
        top_p=0.95,
        context_length=2048,
        system_prompt=(
            "You are an expert code generation assistant. Write clean, "
            "efficient, well-documented code."
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
        max_tokens=1024,
        top_k=40,
        top_p=0.95,
        context_length=2048,
        system_prompt="You are a helpful AI assistant.",
        quantization="int4",
        estimated_vram_mb=1500,
        supports_speculative=False,
    )
