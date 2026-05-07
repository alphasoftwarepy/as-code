"""
AS Code — Application Settings

Pydantic-based settings with .env file support.
All configuration is centralized here.
"""

from __future__ import annotations

import os
import yaml
from functools import lru_cache
from typing import Any, Dict, Optional

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

    _config: Dict[str, Any] = {}

    def __init__(self, **values):
        super().__init__(**values)
        self._load_config()

    def _load_config(self):
        """Load configuration from config.yaml if it exists."""
        config_path = "config.yaml"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading config.yaml: {e}")

    @property
    def models(self) -> Dict[str, Any]:
        """Get model definitions from config."""
        return self._config.get("models", {})

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

    def get_model_path(self, role: str) -> str:
        """Resolve model path from role."""
        model_cfg = self.models.get(role)
        if model_cfg:
            return model_cfg.get("file", "")
        return role


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
