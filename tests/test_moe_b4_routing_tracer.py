"""
AS-Core MoE Engine — Unit Tests for Subfase B4.3.5 (Routing Tracer)
===================================================================
Valida:
1. Funcionamiento correcto del colector de trazas JSONL (RoutingTracer).
2. Estructura exacta de eventos: seq, token, layer, ts, experts (Top-4), weights (sum ~ 1.0).
3. Zero overhead cuando el tracing está desactivado.
4. Persistencia en disco y reproducibilidad de trazas.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from core.moe.cuda_driver import CUDADriver
from core.moe.cublas_backend import CuBLASBackend
from core.moe.expert_registry import ExpertRegistry
from core.moe.router import RealRouter
from core.moe.routing_tracer import RoutingTracer, RoutingTraceEvent

QWEN_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"


@pytest.fixture(scope="module")
def shared_registry():
    return ExpertRegistry(QWEN_PATH)


@pytest.fixture(scope="module")
def shared_cuda_driver():
    driver = CUDADriver()
    yield driver
    driver.destroy()


@pytest.fixture(scope="module")
def shared_cublas(shared_cuda_driver):
    backend = CuBLASBackend(cuda_driver=shared_cuda_driver)
    yield backend
    backend.destroy()


@pytest.fixture
def router(shared_registry, shared_cuda_driver, shared_cublas):
    r = RealRouter(registry=shared_registry, k_active=4, cuda_driver=shared_cuda_driver, cublas_backend=shared_cublas)
    yield r
    r.release()


class TestRoutingTracer:
    """Pruebas unitarias para RoutingTracer (B4.3.5)."""

    def test_b435_tracer_event_format_and_persistence(self, router):
        """Valida que los eventos se formateen y persistan con todos los campos obligatorios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "test_trace.jsonl"
            tracer = RoutingTracer(output_file=trace_path, enabled=True, buffer_flush_size=10)
            tracer.start_sequence(seq_id=0)

            np.random.seed(42)
            x = np.random.randn(1, 2048).astype(np.float32)

            for token_idx in range(5):
                for layer_id in range(3): # Capas 0, 1, 2
                    decision = router.route_token(layer_id=layer_id, x=x, use_gpu=True)
                    tracer.record(token_index=token_idx, layer_id=layer_id, decision=decision)

            tracer.flush()
            assert trace_path.exists()

            with open(trace_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]

            assert len(lines) == 15
            for row in lines:
                assert "seq" in row
                assert "token" in row
                assert "layer" in row
                assert "ts" in row
                assert "experts" in row
                assert "weights" in row
                assert len(row["experts"]) == 4
                assert len(row["weights"]) == 4
                assert abs(sum(row["weights"]) - 1.0) < 1e-5

    def test_b435_zero_overhead_when_disabled(self, router):
        """Valida que cuando el tracer está desactivado no registre ni asigne archivos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "disabled_trace.jsonl"
            tracer = RoutingTracer(output_file=trace_path, enabled=False)

            np.random.seed(42)
            x = np.random.randn(1, 2048).astype(np.float32)
            decision = router.route_token(layer_id=0, x=x, use_gpu=True)

            tracer.record(token_index=0, layer_id=0, decision=decision)
            tracer.flush()

            assert not trace_path.exists()
            assert tracer.total_events_recorded == 0
