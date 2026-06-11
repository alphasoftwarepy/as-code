"""
AS Core — LiteRT Embedded Persistent Engine Provider

Uses the native Python FFI API of litert_lm to run GPU-resident inference,
reusing the existing EngineManager lifecycle and residency policies.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sys
import threading
import time
from typing import AsyncIterator, Optional

# Ensure litert-lm site-packages is in path
LITERT_LM_PATH = r"C:\Users\rva10\AppData\Roaming\uv\tools\litert-lm\Lib\site-packages"
if LITERT_LM_PATH not in sys.path:
    sys.path.append(LITERT_LM_PATH)

try:
    import litert_lm
except ImportError:
    litert_lm = None

from providers.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
    ProviderStatus,
    ProviderType,
)

logger = logging.getLogger("as-code.providers.litert_embedded")


class LiteRTEmbeddedProvider(InferenceProvider):
    """Inference provider using the native embedded litert_lm.Engine."""

    def __init__(self, models_dir: str = "models") -> None:
        super().__init__()
        self._models_dir = models_dir
        self._engine: Optional[litert_lm.Engine] = None
        self._loaded_model_id: Optional[str] = None
        self._lock = threading.Lock()
        self._available = False

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_gpu=True,
            supports_npu=False,
            supports_streaming=True,
            supports_speculative_decoding=False,
            supports_multi_model=False,  # Single-model active policy
            supports_vision=False,
            supports_audio=False,
            max_context_length=2048,
            supported_quantizations=("int4",),
            provider_type=ProviderType.LITERT_NATIVE,
        )

    # ── Lifecycle ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the provider and check if litert_lm is available."""
        self._status = ProviderStatus.INITIALIZING
        if litert_lm is not None:
            self._available = True
            self._status = ProviderStatus.READY
            logger.info("[LITERT-EMBEDDED] Provider initialized successfully. litert_lm is available.")
        else:
            self._available = False
            self._status = ProviderStatus.ERROR
            self._last_error = "litert_lm Python module could not be imported from the configured path."
            logger.error(f"[LITERT-EMBEDDED] {self._last_error}")

    async def shutdown(self) -> None:
        """Release the engine and free GPU memory."""
        self._status = ProviderStatus.SHUTTING_DOWN
        def _exit():
            with self._lock:
                if self._engine is not None:
                    logger.info(f"[LITERT-EMBEDDED] Shutdown: Unloading engine '{self._loaded_model_id}'")
                    try:
                        self._engine.__exit__(None, None, None)
                    except Exception as e:
                        logger.warning(f"Error calling __exit__ on engine: {e}")
                    self._engine = None
                    self._loaded_model_id = None
                    gc.collect()
        await asyncio.to_thread(_exit)
        self._status = ProviderStatus.SHUTDOWN
        logger.info("[LITERT-EMBEDDED] Provider shutdown complete.")

    # ── Model Management ───────────────────────────────────────

    async def load_model(self, model_id: str, model_path: str) -> None:
        if not self._available:
            raise RuntimeError("litert_lm package is not available.")

        t0 = time.time()
        logger.info(f"[MODEL-LOAD] Loading model persist-engine: {model_id} (path: {model_path})")

        def _load():
            with self._lock:
                # Enforce single-model active policy: unload old before loading new
                if self._engine is not None:
                    logger.info(f"[MODEL-SWAP] Auto-unloading active model '{self._loaded_model_id}' before loading '{model_id}'")
                    try:
                        self._engine.__exit__(None, None, None)
                    except Exception as e:
                        logger.warning(f"Error during swap exit: {e}")
                    self._engine = None
                    self._loaded_model_id = None
                    gc.collect()

                try:
                    engine = litert_lm.Engine(model_path, backend=litert_lm.Backend.GPU)
                    engine.__enter__()
                    on_gpu = True
                except Exception as e:
                    logger.warning(f"Failed to load model on GPU: {e}. Falling back to CPU backend.")
                    engine = litert_lm.Engine(model_path, backend=litert_lm.Backend.CPU)
                    engine.__enter__()
                    on_gpu = False

                self._engine = engine
                self._loaded_model_id = model_id
                return on_gpu

        on_gpu = await asyncio.to_thread(_load)
        duration = time.time() - t0
        logger.info(f"[MODEL-LIFECYCLE] Loaded model '{model_id}' in {duration:.2f}s (GPU: {on_gpu})")

    async def unload_model(self, model_id: str) -> None:
        t0 = time.time()
        def _exit():
            with self._lock:
                if self._engine is not None and self._loaded_model_id == model_id:
                    logger.info(f"[MODEL-UNLOAD] Unloading model persistent-engine: {model_id}")
                    try:
                        self._engine.__exit__(None, None, None)
                    except Exception as e:
                        logger.warning(f"Error during exit: {e}")
                    self._engine = None
                    self._loaded_model_id = None
                    gc.collect()
        await asyncio.to_thread(_exit)
        duration = time.time() - t0
        logger.info(f"[MODEL-LIFECYCLE] Unloaded model '{model_id}' in {duration:.2f}s")

    async def is_model_loaded(self, model_id: str) -> bool:
        return self._engine is not None and self._loaded_model_id == model_id

    async def loaded_models(self) -> list[str]:
        if self._engine is not None and self._loaded_model_id:
            return [self._loaded_model_id]
        return []

    # ── Inference ──────────────────────────────────────────────

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        if self._engine is None:
            raise RuntimeError("No model loaded in persistent engine.")

        t0 = time.time()

        messages = self._parse_prompt_to_messages(request.prompt, request.system_prompt)
        if messages:
            history = messages[:-1]
            query = messages[-1]["content"]
        else:
            history = []
            query = request.prompt

        def _run():
            with self._lock:
                if self._engine is None:
                    raise RuntimeError("Model was unloaded while waiting for lock.")
                with self._engine.create_conversation(messages=history) as conv:
                    return conv.send_message(query)

        resp = await asyncio.to_thread(_run)

        text = self._extract_text(resp)
        tokens_gen = resp.get("tokens_generated", 0) if isinstance(resp, dict) else 0
        prompt_tokens = resp.get("prompt_tokens", 0) if isinstance(resp, dict) else 0
        latency = (time.time() - t0) * 1000

        return InferenceResult(
            text=text,
            finish_reason="stop",
            tokens_generated=tokens_gen,
            prompt_tokens=prompt_tokens,
            latency_ms=latency,
            tokens_per_sec=(tokens_gen / (latency / 1000.0)) if latency > 0 and tokens_gen > 0 else 0.0,
            model_id=request.model_id,
            provider_type=ProviderType.LITERT_NATIVE.value,
        )

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[InferenceResult]:
        if self._engine is None:
            yield InferenceResult(
                text="No model loaded in persistent engine.",
                finish_reason="error",
                model_id=request.model_id,
                provider_type=ProviderType.LITERT_NATIVE.value,
            )
            return

        q = asyncio.Queue()
        loop = asyncio.get_running_loop()

        messages = self._parse_prompt_to_messages(request.prompt, request.system_prompt)
        if messages:
            history = messages[:-1]
            query = messages[-1]["content"]
        else:
            history = []
            query = request.prompt

        def thread_worker():
            try:
                with self._lock:
                    if self._engine is None:
                        raise RuntimeError("Model was unloaded while waiting for lock.")
                    with self._engine.create_conversation(messages=history) as conv:
                        for chunk in conv.send_message_async(query):
                            loop.call_soon_threadsafe(q.put_nowait, chunk)
                loop.call_soon_threadsafe(q.put_nowait, None)  # Sentinel for EOF
            except Exception as e:
                logger.error(f"[LITERT-EMBEDDED] Error in streaming thread: {e}")
                loop.call_soon_threadsafe(q.put_nowait, e)

        # Start worker thread
        thread = threading.Thread(target=thread_worker, daemon=True)
        thread.start()

        # Consume queue
        tokens_generated = 0
        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield InferenceResult(
                    text=f"\n[Streaming Error: {item}]",
                    finish_reason="error",
                    model_id=request.model_id,
                    provider_type=ProviderType.LITERT_NATIVE.value,
                )
                break

            text_chunk = self._extract_text(item)
            if text_chunk:
                tokens_generated += 1
                yield InferenceResult(
                    text=text_chunk,
                    finish_reason=None,
                    tokens_generated=tokens_generated,
                    model_id=request.model_id,
                    provider_type=ProviderType.LITERT_NATIVE.value,
                )

        # Final stop chunk
        yield InferenceResult(
            text="",
            finish_reason="stop",
            tokens_generated=tokens_generated,
            model_id=request.model_id,
            provider_type=ProviderType.LITERT_NATIVE.value,
        )

    async def cancel_generation(self, request_id: str) -> None:
        pass  # Cancellation of FFI thread not directly exposed in simple wrapper

    # ── Health & Telemetry ─────────────────────────────────────

    async def health_check(self) -> bool:
        return self._available

    async def get_metrics(self) -> dict:
        return {
            "provider_type": ProviderType.LITERT_NATIVE.value,
            "status": self._status.value,
            "available": self._available,
            "loaded_models": await self.loaded_models(),
        }

    # ── Helpers ────────────────────────────────────────────────

    def _extract_text(self, val) -> str:
        if not val:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            content = val.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts)
            elif isinstance(content, str):
                return content
        return ""

    def _parse_prompt_to_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list[dict]:
        """Reconstruct chat messages array from a compiled prompt string."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        parts = prompt.split("\n\n")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("System: "):
                messages.append({"role": "system", "content": part[len("System: "):]})
            elif part.startswith("User: "):
                messages.append({"role": "user", "content": part[len("User: "):]})
            elif part.startswith("Assistant:"):
                content = part[len("Assistant:"):].strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
            elif part.startswith("Tool Output "):
                idx = part.find("): ")
                if idx != -1:
                    content = part[idx + 3:]
                    messages.append({"role": "tool", "content": content})
                else:
                    messages.append({"role": "tool", "content": part})
        return messages
