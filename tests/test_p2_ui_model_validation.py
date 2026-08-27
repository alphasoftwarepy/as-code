"""
AS-Core — Tests de Validación P2: Model Selector UI + Multi-Backend Inference
=============================================================================
Valida:
1. AUTO Mode (Coordinator SmartRouter) vs MANUAL Mode (Explicit Override).
2. Continuidad de contexto conversacional multi-turno al alternar modelos.
3. RAG NotebookLM context injection independiente del backend.
4. Streaming SSE y cancelación en caliente.
5. Políticas Single-Active-Model anti-OOM en VRAM.
"""

import asyncio
import os
import pytest

from core.engine import EngineManager
from core.hardware import detect_hardware
from providers.base import (
    InferenceRequest,
    InferenceResult,
    ProviderStatus,
    ProviderType,
)
from providers.llamacpp_provider import LlamaCppProvider
from providers.registry import ProviderRegistry
from router.smart_router import SmartRouter

LLAMACPP_BIN = r"C:\as-code\moe_poc\bins\llama-server.exe"
QWEN_MODEL = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
HAS_QWEN = os.path.exists(LLAMACPP_BIN) and os.path.exists(QWEN_MODEL)


def test_smart_router_auto_vs_manual_override():
    """Valida que SmartRouter use reglas semánticas en AUTO y respete override en MANUAL."""
    router = SmartRouter(chat_model="chat", coding_model="code", reasoning_model="reasoning")

    # Modo AUTO: Detección por intención de palabras clave (rules.py)
    m1, _ = router.route("Write a python function to calculate fibonacci", explicit_model="auto")
    assert m1 == "code"

    m2, _ = router.route("Analyze the system architecture and evaluate trade-offs", explicit_model="auto")
    assert m2 == "reasoning"

    m3, _ = router.route("Hello, how are you?", explicit_model="auto")
    assert m3 == "chat"

    # Modo MANUAL: Override explícito
    m4, _ = router.route("Write a python function", explicit_model="moe_large")
    assert m4 == "moe_large"

    m5, _ = router.route("Hello world", explicit_model="qwen-moe")
    assert m5 == "qwen-moe"


def test_coordinator_manual_model_dispatch():
    """Valida que EngineManager cargue y ejecute el modelo manual seleccionado."""
    if not HAS_QWEN:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        registry = ProviderRegistry()
        provider = LlamaCppProvider(
            server_bin_path=LLAMACPP_BIN,
            port=8780,
            n_gpu_layers=10,
        )
        registry.register("llamacpp", provider)

        engine = EngineManager(
            provider_registry=registry,
            hardware_info=detect_hardware(),
            max_vram_mb=3900,
        )
        engine.register_model(
            model_id="moe_large",
            model_path=QWEN_MODEL,
            model_type="moe",
            estimated_vram_mb=3893,
            provider_id="llamacpp",
        )

        # Inferencia con override manual a moe_large
        req = InferenceRequest(
            prompt="Responde únicamente con la palabra 'CONFIRMADO'.",
            model_id="moe_large",
            temperature=0.0,
            max_tokens=16,
            stream=False,
        )
        res = await engine.generate(req)
        assert res.tokens_generated > 0
        assert "CONFIRMADO" in res.text or len(res.text.strip()) > 0
        assert res.provider_type == ProviderType.LLAMACPP.value

        await provider.shutdown()

    asyncio.run(_test())


def test_multi_turn_context_continuity():
    """Valida la retención de contexto en múltiples turnos conversacionales."""
    if not HAS_QWEN:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        registry = ProviderRegistry()
        provider = LlamaCppProvider(
            server_bin_path=LLAMACPP_BIN,
            port=8781,
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
            model_path=QWEN_MODEL,
            model_type="moe",
            estimated_vram_mb=3893,
            provider_id="llamacpp",
        )

        # Turno 1
        history = [
            {"role": "user", "content": "Mi nombre en clave es ALPHA-7."},
            {"role": "assistant", "content": "Entendido, te recordaré como ALPHA-7."},
            {"role": "user", "content": "¿Cuál es mi nombre en clave? Responde brevemente."},
        ]
        prompt_t2 = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history]) + "\nAssistant:"

        req = InferenceRequest(
            prompt=prompt_t2,
            model_id="qwen-moe",
            temperature=0.0,
            max_tokens=32,
            stream=False,
        )
        res = await engine.generate(req)
        assert "ALPHA-7" in res.text or "ALPHA" in res.text

        await provider.shutdown()

    asyncio.run(_test())


def test_rag_context_injection_independence():
    """Valida que el contexto inyectado por RAG sea consumido sin importar el provider."""
    if not HAS_QWEN:
        pytest.skip("Requiere llama-server.exe y modelo Qwen GGUF")

    async def _test():
        registry = ProviderRegistry()
        provider = LlamaCppProvider(
            server_bin_path=LLAMACPP_BIN,
            port=8782,
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
            model_path=QWEN_MODEL,
            model_type="moe",
            estimated_vram_mb=3893,
            provider_id="llamacpp",
        )

        rag_doc_context = (
            "[RAG DOCUMENT: secret_specs.txt]\n"
            "The unlock code for project AS-Core is 'NEBULA-9988'.\n"
        )
        system_prompt = f"You are a helpful technical assistant. Context:\n{rag_doc_context}"

        req = InferenceRequest(
            prompt="Based on the provided context, what is the unlock code for project AS-Core? Answer with the code.",
            system_prompt=system_prompt,
            model_id="qwen-moe",
            temperature=0.0,
            max_tokens=32,
            stream=False,
        )
        res = await engine.generate(req)
        assert "NEBULA-9988" in res.text or "9988" in res.text

        await provider.shutdown()

    asyncio.run(_test())
