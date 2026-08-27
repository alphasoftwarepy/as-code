"""
Tests for LlamaCppProvider — Windows CUDA Subprocess Integration (P1)
====================================================================
"""

import asyncio
import os
import pytest

from core.engine import EngineManager
from core.hardware import detect_hardware
from providers.base import (
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
    ProviderStatus,
    ProviderType,
)
from providers.llamacpp_provider import LlamaCppProvider
from providers.registry import ProviderRegistry

# Rutas estándar del modelo de prueba y binario
BIN_PATH = r"C:\as-code\moe_poc\bins\llama-server.exe"
MODEL_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
HAS_LLAMACPP = os.path.exists(BIN_PATH) and os.path.exists(MODEL_PATH)


def test_llamacpp_capabilities_and_initialization():
    """Verifica capacidades y ciclo inicial de LlamaCppProvider (Unit Test)."""
    async def _test():
        provider = LlamaCppProvider(server_bin_path=BIN_PATH, port=8766)
        caps = provider.capabilities()

        assert caps.provider_type == ProviderType.LLAMACPP
        assert caps.supports_gpu is True
        assert caps.supports_streaming is True
        assert caps.max_context_length == 2048
        assert "q4_k_m" in caps.supported_quantizations

        await provider.initialize()
        if os.path.exists(BIN_PATH):
            assert provider.status == ProviderStatus.READY
        else:
            assert provider.status == ProviderStatus.ERROR

    asyncio.run(_test())


def test_llamacpp_process_lifecycle_and_health():
    """Valida el ciclo de vida del proceso daemon y endpoint /health (Hardware Integration Test)."""
    if not HAS_LLAMACPP:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        provider = LlamaCppProvider(
            server_bin_path=BIN_PATH,
            port=8767,
            n_gpu_layers=10,
        )
        await provider.initialize()
        assert provider.status == ProviderStatus.READY

        # 1. Cargar modelo
        await provider.load_model("qwen-moe", MODEL_PATH)
        assert await provider.is_model_loaded("qwen-moe") is True
        assert await provider.health_check() is True
        assert "qwen-moe" in await provider.loaded_models()

        metrics = await provider.get_metrics()
        assert metrics["backend"] == "llamacpp"
        assert metrics["loaded"] is True
        assert metrics["process_pid"] is not None

        # 2. Descargar modelo
        await provider.unload_model("qwen-moe")
        assert await provider.is_model_loaded("qwen-moe") is False
        assert await provider.health_check() is False
        assert await provider.loaded_models() == []

        # 3. Shutdown
        await provider.shutdown()
        assert provider.status == ProviderStatus.SHUTDOWN

    asyncio.run(_test())


def test_llamacpp_atomic_generate():
    """Valida inferencia atómica no-streaming (Hardware Integration Test)."""
    if not HAS_LLAMACPP:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        provider = LlamaCppProvider(
            server_bin_path=BIN_PATH,
            port=8768,
            n_gpu_layers=10,
        )
        await provider.initialize()
        await provider.load_model("qwen-moe", MODEL_PATH)

        try:
            req = InferenceRequest(
                prompt="Calcula 2 + 2 y responde únicamente con el número.",
                model_id="qwen-moe",
                temperature=0.0,
                max_tokens=32,
                stream=False,
            )
            result = await provider.generate(req)

            assert isinstance(result, InferenceResult)
            assert result.provider_type == ProviderType.LLAMACPP.value
            assert result.tokens_generated > 0
            assert result.tokens_per_sec > 0.0
            assert "4" in result.text
        finally:
            await provider.shutdown()

    asyncio.run(_test())


def test_llamacpp_streaming_generation():
    """Valida inferencia reactiva SSE por tokens (Hardware Integration Test)."""
    if not HAS_LLAMACPP:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        provider = LlamaCppProvider(
            server_bin_path=BIN_PATH,
            port=8769,
            n_gpu_layers=10,
        )
        await provider.initialize()
        await provider.load_model("qwen-moe", MODEL_PATH)

        try:
            req = InferenceRequest(
                prompt="Cuenta del 1 al 5.",
                model_id="qwen-moe",
                temperature=0.0,
                max_tokens=32,
                stream=True,
            )

            tokens = []
            async for chunk in provider.generate_stream(req):
                if chunk.text:
                    tokens.append(chunk.text)

            full_text = "".join(tokens)
            assert len(tokens) > 1
            assert len(full_text.strip()) > 0
        finally:
            await provider.shutdown()

    asyncio.run(_test())


def test_llamacpp_cancellation():
    """Valida interrupción segura de stream (Hardware Integration Test)."""
    if not HAS_LLAMACPP:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        provider = LlamaCppProvider(
            server_bin_path=BIN_PATH,
            port=8770,
            n_gpu_layers=10,
        )
        await provider.initialize()
        await provider.load_model("qwen-moe", MODEL_PATH)

        try:
            req_id = "test-cancel-123"
            req = InferenceRequest(
                prompt="Escribe un ensayo de 500 palabras sobre la historia de la computación.",
                model_id="qwen-moe",
                temperature=0.7,
                max_tokens=500,
                stream=True,
                request_id=req_id,
            )

            received_chunks = 0
            async for chunk in provider.generate_stream(req):
                received_chunks += 1
                if received_chunks == 3:
                    await provider.cancel_generation(req_id)

            assert received_chunks < 50
        finally:
            await provider.shutdown()

    asyncio.run(_test())


def test_llamacpp_engine_manager_e2e():
    """Valida la integración completa de LlamaCppProvider con EngineManager."""
    if not HAS_LLAMACPP:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        registry = ProviderRegistry()
        provider = LlamaCppProvider(
            server_bin_path=BIN_PATH,
            port=8771,
            n_gpu_layers=10,
        )
        registry.register("llamacpp", provider)

        engine = EngineManager(
            provider_registry=registry,
            hardware_info=detect_hardware(),
            max_vram_mb=3900,
        )
        engine.register_model(
            model_id="qwen-moe",
            model_path=MODEL_PATH,
            model_type="moe",
            estimated_vram_mb=3893,
            provider_id="llamacpp",
        )

        try:
            req = InferenceRequest(
                prompt="Hola mundo, responde 'OK'.",
                model_id="qwen-moe",
                temperature=0.0,
                max_tokens=16,
            )

            result_chunks = []
            async for chunk in engine.generate_stream(req):
                if chunk.text:
                    result_chunks.append(chunk.text)

            full_output = "".join(result_chunks)
            assert len(result_chunks) > 0
            assert len(full_output.strip()) > 0
        finally:
            await provider.shutdown()

    asyncio.run(_test())
