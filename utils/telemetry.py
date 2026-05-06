"""
AS Code — Telemetry Collector

Lightweight telemetry for RAM/VRAM usage, tokens/sec, and latency.
All data stays local — zero cloud analytics.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.hardware import get_ram_available_mb, get_vram_free_mb

logger = logging.getLogger("as-code.utils.telemetry")


@dataclass
class InferenceMetric:
    """Single inference measurement."""
    model_id: str
    tokens_generated: int
    tokens_per_sec: float
    latency_ms: float
    first_token_ms: float
    timestamp: float = field(default_factory=time.time)


class TelemetryCollector:
    """Collects and exposes inference metrics.

    Stores last N measurements in a ring buffer for
    lightweight memory usage. No persistence.
    """

    def __init__(self, max_history: int = 100) -> None:
        self._history: deque[InferenceMetric] = deque(maxlen=max_history)
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._start_time: float = time.time()

    def record(self, metric: InferenceMetric) -> None:
        """Record an inference measurement."""
        self._history.append(metric)
        self._total_requests += 1
        self._total_tokens += metric.tokens_generated

    def get_snapshot(self) -> dict:
        """Get current telemetry snapshot."""
        uptime = time.time() - self._start_time

        # Compute averages from recent history
        recent = list(self._history)
        avg_tps = 0.0
        avg_latency = 0.0
        avg_first_token = 0.0

        if recent:
            avg_tps = sum(m.tokens_per_sec for m in recent) / len(recent)
            avg_latency = sum(m.latency_ms for m in recent) / len(recent)
            avg_first_token = sum(m.first_token_ms for m in recent) / len(recent)

        return {
            "uptime_sec": round(uptime, 1),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "avg_tokens_per_sec": round(avg_tps, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "avg_first_token_ms": round(avg_first_token, 1),
            "ram_available_mb": get_ram_available_mb(),
            "vram_free_mb": get_vram_free_mb(),
        }

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get last N inference metrics."""
        recent = list(self._history)[-n:]
        return [
            {
                "model": m.model_id,
                "tokens": m.tokens_generated,
                "tps": round(m.tokens_per_sec, 1),
                "latency_ms": round(m.latency_ms, 1),
                "first_token_ms": round(m.first_token_ms, 1),
            }
            for m in recent
        ]
