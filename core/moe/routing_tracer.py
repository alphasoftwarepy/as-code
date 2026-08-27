"""
AS-Core MoE Engine — Routing Tracer (B4.3.5)
============================================
Registrador de alta precisión y bajo overhead para capturar el comportamiento
de enrutamiento real (Routing Set y Pesos) por capa y por token durante inferencia.
Genera registros estructurados en formato JSONL para análisis offline del working set.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.moe.residency_manager import ResidencyLayerDispatch
from core.moe.router import RoutingDecision

logger = logging.getLogger("as-code.core.moe.tracer")


@dataclass
class RoutingTraceEvent:
    """Evento individual de enrutamiento capturado para un token en una capa."""
    seq: int
    token: int
    layer: int
    ts: int  # Timestamp en nanosegundos para orden temporal estricto
    experts: List[int]
    weights: List[float]
    hits: Optional[List[bool]] = None
    slots: Optional[List[int]] = None

    def to_json_dict(self) -> Dict[str, Any]:
        d = {
            "seq": self.seq,
            "token": self.token,
            "layer": self.layer,
            "ts": self.ts,
            "experts": self.experts,
            "weights": [round(w, 6) for w in self.weights],
        }
        if self.hits is not None:
            d["hits"] = self.hits
        if self.slots is not None:
            d["slots"] = self.slots
        return d


class RoutingTracer:
    """Colector de trazas de enrutamiento con buffer en memoria y persistencia JSONL."""

    def __init__(
        self,
        output_file: Optional[Union[str, Path]] = None,
        enabled: bool = False,
        buffer_flush_size: int = 1000,
    ):
        self.enabled = enabled
        self.output_path = Path(output_file or r"C:\as-code\moe_poc\data\routing_trace.jsonl")
        self.buffer_flush_size = buffer_flush_size
        self._buffer: List[RoutingTraceEvent] = []
        self._current_seq_id: int = 0
        self.total_events_recorded: int = 0
        self.total_trace_time_ns: int = 0

        if self.enabled:
            self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def enable(self) -> None:
        self.enabled = True
        self._ensure_output_dir()

    def disable(self) -> None:
        self.flush()
        self.enabled = False

    def start_sequence(self, seq_id: int, reset_file: bool = False) -> None:
        """Inicia una nueva secuencia de generación."""
        self._current_seq_id = seq_id
        if reset_file and self.output_path.exists():
            try:
                self.output_path.unlink()
            except Exception:
                pass

    def record(
        self,
        token_index: int,
        layer_id: int,
        decision: RoutingDecision,
        residency_dispatch: Optional[ResidencyLayerDispatch] = None,
        seq_id: Optional[int] = None,
    ) -> None:
        """Registra un evento de enrutamiento. Si enabled=False, retorna inmediatamente con overhead cero."""
        if not self.enabled:
            return

        t0 = time.perf_counter_ns()

        hits = None
        slots = None
        if residency_dispatch is not None:
            hits = [rd.is_hit for rd in residency_dispatch.residency_decisions]
            slots = [rd.slot_id for rd in residency_dispatch.residency_decisions]

        event = RoutingTraceEvent(
            seq=seq_id if seq_id is not None else self._current_seq_id,
            token=token_index,
            layer=layer_id,
            ts=t0,
            experts=list(decision.top_k_ids),
            weights=list(decision.normalized_weights),
            hits=hits,
            slots=slots,
        )

        self._buffer.append(event)
        self.total_events_recorded += 1

        if len(self._buffer) >= self.buffer_flush_size:
            self.flush()

        self.total_trace_time_ns += (time.perf_counter_ns() - t0)

    def flush(self) -> None:
        """Escribe todos los eventos del buffer a disco en formato JSONL."""
        if not self._buffer:
            return

        self._ensure_output_dir()
        with open(self.output_path, "a", encoding="utf-8") as f:
            for ev in self._buffer:
                f.write(json.dumps(ev.to_json_dict(), separators=(",", ":")) + "\n")
        self._buffer.clear()

    @property
    def average_tracing_overhead_us(self) -> float:
        """Overhead promedio por evento registrado en microsegundos."""
        if self.total_events_recorded == 0:
            return 0.0
        return (self.total_trace_time_ns / self.total_events_recorded) / 1000.0

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
