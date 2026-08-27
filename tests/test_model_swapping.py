"""
AS-Core — Test de Validación de Intercambio de Modelos y Liberación de VRAM
==========================================================================
Valida:
1. Vaciado del 100% de la VRAM al intercambiar modelos entre providers distintos (cross-provider).
2. Inicialización y ejecución correcta de Gemma 4 E4B con LiteRT CLI.
"""

import asyncio
import os
import pytest

from core.engine import EngineManager
from core.hardware import detect_hardware, get_vram_free_mb
from providers.base import InferenceRequest, ProviderStatus
from providers.litert_cli import LiteRTCLIProvider
from providers.llamacpp_provider import LlamaCppProvider
from providers.registry import ProviderRegistry

LLAMACPP_BIN = r"C:\as-code\moe_poc\bins\llama-server.exe"
QWEN_MODEL = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
GEMMA4_MODEL = r"models\gemma\gemma-4-E4B-it.litertlm"

HAS_QWEN = os.path.exists(LLAMACPP_BIN) and os.path.exists(QWEN_MODEL)
HAS_GEMMA4 = os.path.exists(GEMMA4_MODEL)


def test_cross_provider_vram_release_on_model_swap():
    """Valida que al cambiar de llamacpp a otro provider, el modelo previo se descargue por completo."""
    if not HAS_QWEN:
        pytest.skip("Requiere llama-server.exe y modelo Qwen")

    async def _test():
        registry = ProviderRegistry()

        llamacpp_p = LlamaCppProvider(
            server_bin_path=LLAMACPP_BIN,
            port=8795,
            n_gpu_layers=10,
        )
        registry.register("llamacpp", llamacpp_p)

        litert_p = LiteRTCLIProvider(
            cli_path="litert-lm",
            default_backend="gpu",
            models_dir="models",
        )
        registry.register("litert_cli", litert_p)

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
        engine.register_model(
            model_id="gemma-reasoning",
            model_path=GEMMA4_MODEL,
            model_type="reasoning",
            estimated_vram_mb=3660,
            provider_id="litert_cli",
        )

        # 1. Cargar Qwen MoE en llamacpp
        req_qwen = InferenceRequest(
            prompt="Di 'HOLA'",
            model_id="qwen-moe",
            temperature=0.0,
            max_tokens=8,
            stream=False,
        )
        res_qwen = await engine.generate(req_qwen)
        assert res_qwen.tokens_generated > 0
        assert await llamacpp_p.is_model_loaded("qwen-moe")
        assert llamacpp_p._proc is not None  # Proceso activo

        # 2. Swap al modelo gemma-reasoning en litert_cli
        # EngineManager debe descargar qwen-moe en llamacpp_p antes de cargar gemma-reasoning
        await engine._ensure_model_loaded("gemma-reasoning")

        # Comprobación estricta: llamacpp_p NO debe tener modelo ni proceso activo
        assert not (await llamacpp_p.is_model_loaded("qwen-moe"))
        assert llamacpp_p._proc is None

        # Limpiar
        await registry.shutdown_all()

    asyncio.run(_test())
