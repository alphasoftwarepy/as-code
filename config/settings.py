"""
AS Code — Application Settings

Pydantic-based settings with .env file support.
All configuration is centralized here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ── Server ─────────────────────────────────────────────────
    host: str = Field(default="127.0.0.1", description="API server host")
    port: int = Field(default=8000, description="API server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # ── Models ─────────────────────────────────────────────────
    models_dir: str = Field(
        default="models", description="Directory for model files"
    )

    reasoning_model_id: str = Field(
        default="gemma-3n-web",
        description="General conversation model identifier"
    )

    reasoning_model_file: str = Field(
        default="models/gemma/gemma-3n-E2B-it-int4.litertlm",
        description="General conversation model filename",
    )

    reasoning_model_vram_mb: int = Field(
        default=1500,
        description="Estimated VRAM for reasoning model (MB)"
    )

    coding_model_id: str = Field(
        default="gemma-3n-code",
        description="Coding model identifier"
    )

    coding_model_file: str = Field(
        default="models/gemma/gemma-3n-E2B-it-int4.litertlm",
        description="Coding model filename",
    )

    coding_model_vram_mb: int = Field(
        default=1500,
        description="Estimated VRAM for coding model (MB)"
    )

    # ── Provider ───────────────────────────────────────────────
    active_provider: str = Field(
        default="litert_cli", description="Active inference provider"
    )
    litert_cli_path: Optional[str] = Field(
        default=None, description="Path to litert-lm CLI (auto-detected)"
    )
    litert_backend: str = Field(
        default="gpu", description="LiteRT backend (gpu/cpu)"
    )
    enable_speculative_decoding: bool = Field(
        default=True, description="Enable speculative decoding for Gemma 4"
    )

    # ── Inference ──────────────────────────────────────────────
    default_temperature: float = Field(
        default=0.7, description="Default temperature"
    )
    default_max_tokens: int = Field(
        default=1024, description="Default max tokens"
    )
    max_context_length: int = Field(
        default=2048, description="Maximum context length"
    )

    # ── Hardware Adaptive ──────────────────────────────────────
    max_vram_usage_mb: int = Field(
        default=3200, description="Maximum VRAM usage in MB"
    )
    model_unload_timeout_sec: float = Field(
        default=300.0, description="Seconds before unloading idle model"
    )
    anti_oom_threshold_mb: int = Field(
        default=500, description="Minimum free RAM before warning (MB)"
    )

    # ── System Mode ────────────────────────────────────────────
    system_mode: str = Field(
        default="balanced",
        description="System mode: ultra_light, balanced, performance",
    )

    model_config = {
        "env_file": ".env",
        "env_prefix": "ASCODE_",
        "case_sensitive": False,
    }

    def get_model_path(self, model_id: str) -> str:
        """Resolve LiteRT model registry reference."""

        if model_id == self.reasoning_model_id:
            return self.reasoning_model_file

        elif model_id == self.coding_model_id:
            return self.coding_model_file

        return model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
