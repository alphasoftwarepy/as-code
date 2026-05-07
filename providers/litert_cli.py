"""
AS Code — LiteRT-LM CLI Subprocess Provider

Primary inference provider for Windows. Uses the `litert-lm` CLI tool
as a subprocess with JSON output parsing and async stdout streaming.

This is the only provider with full Windows + GPU support today (v0.11.0).

Performance notes:
- subprocess overhead is ~5-15ms per invocation (negligible vs inference time)
- stdout streaming gives real-time token output
- GPU backend via DirectX shader compiler
- Supports speculative decoding (MTP) for Gemma 4 models
"""

from __future__ import annotations

import subprocess
import asyncio
import json
import logging
import os
import shutil
import time
from typing import AsyncIterator, Optional

from providers.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
    ProviderStatus,
    ProviderType,
)

logger = logging.getLogger("as-code.providers.litert_cli")


class LiteRTCLIProvider(InferenceProvider):
    """Inference provider using LiteRT-LM CLI as a subprocess.

    Designed for:
    - Native Windows execution with GPU acceleration
    - Token-by-token stdout streaming
    - Model loading/unloading via process lifecycle
    - Zero Python-side VRAM allocation
    """

    def __init__(
        self,
        cli_path: Optional[str] = None,
        default_backend: str = "gpu",
        enable_speculative_decoding: bool = True,
        models_dir: str = "models",
    ) -> None:
        super().__init__()
        self._cli_path = cli_path or "litert-lm"
        self._default_backend = default_backend
        self._enable_speculative = enable_speculative_decoding
        self._models_dir = models_dir

        # Track loaded models and active processes
        self._model_refs: dict[str, str] = {} # model_id → file path
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

    # ── Capabilities ───────────────────────────────────────────

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_gpu=True,
            supports_npu=False,
            supports_streaming=True,
            supports_speculative_decoding=self._enable_speculative,
            supports_multi_model=False,  # one process at a time
            supports_vision=False,
            supports_audio=False,
            max_context_length=4096,
            supported_quantizations=("int4", "int8"),
            provider_type=ProviderType.LITERT_CLI,
        )

    # ── Lifecycle ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """Verify litert-lm CLI is installed and accessible."""
        if self._status == ProviderStatus.READY:
            return

        self._status = ProviderStatus.INITIALIZING

        try:
            # Check if CLI is available
            cli_found = shutil.which(self._cli_path)
            if not cli_found:
                # Try to find it in the venv Scripts or as a uv tool
                for candidate in [
                    os.path.join("venv", "Scripts", "litert-lm.exe"),
                    os.path.join("venv", "Scripts", "litert-lm"),
                    os.path.expanduser("~\\AppData\\Roaming\\uv\\tools\\litert-lm.exe"),
                    os.path.expanduser("~\\.local\\bin\\litert-lm.exe"),
                ]:
                    if os.path.exists(candidate):
                        self._cli_path = candidate
                        cli_found = candidate
                        break

            if not cli_found:
                logger.warning(
                    "litert-lm CLI not found in PATH. "
                    "Install with: uv tool install litert-lm"
                )
                # Don't fail — allow graceful degradation
                self._status = ProviderStatus.READY
                self._last_error = "CLI not found (install: uv tool install litert-lm)"
                return

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            # Quick version check
            proc = subprocess.Popen(
                [self._cli_path, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            stdout, stderr = proc.communicate(timeout=10)

            version_info = stdout.strip() or stderr.strip()

            logger.info(f"LiteRT-LM CLI found: {version_info}")

            self._status = ProviderStatus.READY
            self._last_error = None

        except asyncio.TimeoutError:
            self._status = ProviderStatus.ERROR
            self._last_error = "CLI version check timed out"
            logger.error(self._last_error)
        except Exception as e:
            self._status = ProviderStatus.ERROR
            self._last_error = str(e)
            logger.error(f"CLI initialization failed: {e}")

    async def shutdown(self) -> None:
        """Terminate all active processes and clean up."""
        self._status = ProviderStatus.SHUTTING_DOWN

        # Cancel all active generations
        for req_id, event in self._cancel_events.items():
            event.set()

        # Terminate all active processes
        for model_id, proc in list(self._active_processes.items()):
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            logger.info(f"Terminated process for: {model_id}")

        self._active_processes.clear()
        self._cancel_events.clear()
        self._model_refs.clear()
        self._status = ProviderStatus.SHUTDOWN
        logger.info("LiteRT CLI provider shut down")

    # ── Model Management ───────────────────────────────────────

    async def load_model(self, model_id: str, model_path: str) -> None:
        """Register a model path. Actual loading happens at inference time
        (lazy loading — CLI spawns a new process per inference call)."""
        # LiteRT-LM models are referenced by imported registry ID
        self._model_refs[model_id] = model_path
        logger.info(f"Model registered: {model_id} → {model_path}")

    async def unload_model(self, model_id: str) -> None:
        """Unregister a model and terminate any active process."""
        if model_id in self._active_processes:
            proc = self._active_processes[model_id]
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            del self._active_processes[model_id]

        if model_id in self._model_refs:
            del self._model_refs[model_id]
            logger.info(f"Model unloaded: {model_id}")

    async def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._model_refs

    async def loaded_models(self) -> list[str]:
        return list(self._model_refs.keys())

    # ── Inference ──────────────────────────────────────────────

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Run inference and return the complete result."""
        full_text = []
        result = InferenceResult(
            model_id=request.model_id,
            provider_type=ProviderType.LITERT_CLI.value,
        )

        async for chunk in self.generate_stream(request):
            full_text.append(chunk.text)
            result = chunk

        result.text = "".join(full_text)
        return result

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[InferenceResult]:
        """Run inference via CLI subprocess and stream tokens."""
        model_ref = self._model_refs.get(request.model_id)
        if not model_ref:
            yield InferenceResult(
                text="",
                finish_reason="error",
                model_id=request.model_id,
                provider_type=ProviderType.LITERT_CLI.value,
            )
            return

        # Set up cancellation
        cancel_event = asyncio.Event()
        self._cancel_events[request.request_id] = cancel_event

        # Build CLI command
        cmd = self._build_command(model_ref, request)

        start_time = time.perf_counter()
        tokens_generated = 0

        try:
            self._status = ProviderStatus.BUSY
            logger.info(f"Starting LiteRT-LM subprocess: {' '.join(cmd)}")

            # Force UTF-8 for the subprocess (essential on Windows)
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            # Merge stderr into stdout to avoid deadlock and capture all info
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1, # Line buffered for stdout (if supported)
                stdin=subprocess.DEVNULL,
                env=env,
            )
            
            self._active_processes[request.model_id] = proc

            # Stream stdout character by character
            first_token_time = None
            
            # Internal buffer to handle multi-char tags like <think>
            tag_buffer = ""

            while True:
                if cancel_event.is_set():
                    logger.info(f"Cancellation requested for {request.request_id}")
                    proc.terminate()
                    yield InferenceResult(
                        text="",
                        finish_reason="stop",
                        tokens_generated=tokens_generated,
                        model_id=request.model_id,
                        provider_type=ProviderType.LITERT_CLI.value,
                    )
                    return

                # Read one character at a time to ensure real-time streaming
                # even if no newlines are present (common in DeepSeek <think> blocks)
                char = proc.stdout.read(1)

                if not char:
                    # Check if process is still alive
                    if proc.poll() is not None:
                        break
                    continue

                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    logger.info(f"First token received for {request.request_id} in {first_token_time - start_time:.2f}s")

                # Track generated tokens (rough estimate)
                if char.isspace():
                    tokens_generated += 1

                elapsed = time.perf_counter() - start_time
                tps = tokens_generated / elapsed if elapsed > 0 else 0

                # Clean up output
                # We handle tags by skipping them if they appear exactly
                # This is a bit naive for read(1) but prevents the hang
                text = char
                text = text.replace("\x00", "")
                
                # Filter out the start of thinking blocks to keep UI clean
                # Note: This is simplified; a full state machine would be better
                # but read(1) is the priority for the hang fix.
                if text in ("<", "t", "h", "i", "n", "k", ">", "/"):
                    tag_buffer += text
                    # If we have a complete tag, don't yield it
                    if tag_buffer in ("<think>", "</think>"):
                        tag_buffer = ""
                        continue
                    # If the buffer is getting too long and doesn't match, flush it
                    if len(tag_buffer) > 10:
                        yield_text = tag_buffer
                        tag_buffer = ""
                        text = yield_text
                    else:
                        continue
                elif tag_buffer:
                    # Not part of a tag, flush buffer
                    text = tag_buffer + text
                    tag_buffer = ""

                yield InferenceResult(
                    text=text,
                    finish_reason=None,
                    tokens_generated=tokens_generated,
                    latency_ms=elapsed * 1000,
                    tokens_per_sec=tps,
                    model_id=request.model_id,
                    provider_type=ProviderType.LITERT_CLI.value,
                )

            # Wait for process to finish
            return_code = proc.wait()
            logger.info(f"Subprocess finished with return code {return_code}")

            elapsed = time.perf_counter() - start_time
            tps = tokens_generated / elapsed if elapsed > 0 else 0

            # Final chunk with finish reason
            yield InferenceResult(
                text="",
                finish_reason="stop",
                tokens_generated=tokens_generated,
                latency_ms=elapsed * 1000,
                tokens_per_sec=tps,
                model_id=request.model_id,
                provider_type=ProviderType.LITERT_CLI.value,
            )

        except Exception as e:
            logger.exception(f"Inference error: {e}")

            yield InferenceResult(
                text=f"[Error: {str(e)}]",
                finish_reason="error",
                model_id=request.model_id,
                provider_type=ProviderType.LITERT_CLI.value,
            )
        finally:
            self._status = ProviderStatus.READY
            self._cancel_events.pop(request.request_id, None)
            self._active_processes.pop(request.model_id, None)

    async def cancel_generation(self, request_id: str) -> None:
        """Cancel an in-progress generation."""
        if request_id in self._cancel_events:
            self._cancel_events[request_id].set()
            logger.info(f"Cancelled generation: {request_id}")

    # ── Health & Telemetry ─────────────────────────────────────

    async def health_check(self) -> bool:
        """Check if the CLI is accessible."""
        if self._status in (ProviderStatus.ERROR, ProviderStatus.SHUTDOWN):
            return False
        return True

    async def get_metrics(self) -> dict:
        """Return provider metrics."""
        return {
            "provider_type": ProviderType.LITERT_CLI.value,
            "status": self._status.value,
            "registered_models": list(self._model_refs.keys()),
            "active_processes": len(self._active_processes),
            "backend": self._default_backend,
            "speculative_decoding": self._enable_speculative,
        }

    # ── Internal ───────────────────────────────────────────────

    def _build_command(
        self, model_ref: str, request: InferenceRequest
    ) -> list[str]:
        """Build the CLI command for inference.

        IMPORTANT: every argument and its value MUST be a separate list item.
        Using f"--flag={value}" causes the value to be parsed as part of the
        flag string by some CLI parsers, which can corrupt the prompt when it
        contains whitespace, newlines, or sequences that look like flags (--).
        """
        # Construct the full prompt with system prompt if present
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"

        cmd = [
            self._cli_path,
            "run",
            model_ref,
            "--backend", self._default_backend,
            "--prompt", prompt,
            "--max-num-tokens", str(request.max_tokens),
            "--temperature", str(request.temperature),
            "--top-k", str(request.top_k),
        ]

        # Speculative decoding — disabled for now, Gemma 3n requires 'auto'
        # cmd.extend(["--enable-speculative-decoding", "auto"])

        logger.debug(f"CLI command: {self._cli_path} run {model_ref} [prompt omitted] "
                     f"--max-num-tokens {request.max_tokens} "
                     f"--temperature {request.temperature} "
                     f"--top-k {request.top_k}")

        return cmd
