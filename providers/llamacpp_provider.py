"""
AS Code — llama.cpp Windows CUDA Provider (P1)
==============================================
Adaptador de inferencia para llama.cpp nativo sobre Windows con aceleración CUDA.
Ejecuta `llama-server.exe` como subproceso daemon y expone la API OpenAI-compatible
cumpliendo estrictamente el contrato abstracto `InferenceProvider`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, AsyncIterator, Dict, List, Optional

from providers.base import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
    ProviderStatus,
    ProviderType,
)

logger = logging.getLogger("as-code.providers.llamacpp")


def _is_port_in_use(host: str, port: int) -> bool:
    """Comprueba si un puerto TCP local está actualmente ocupado."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False


def _find_available_port(host: str, start_port: int, max_attempts: int = 25) -> int:
    """Busca dinámicamente un puerto libre a partir de start_port."""
    for p in range(start_port, start_port + max_attempts):
        if not _is_port_in_use(host, p):
            return p
    return start_port


class LlamaCppProvider(InferenceProvider):
    """Proveedor de inferencia para llama.cpp en Windows CUDA."""

    def __init__(
        self,
        server_bin_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8766,
        n_gpu_layers: int = 10,
        context_size: int = 2048,
        startup_timeout_sec: float = 90.0,
    ) -> None:
        super().__init__()
        self._server_bin_path = server_bin_path or r"C:\as-code\moe_poc\bins\llama-server.exe"
        self._host = host
        self._port = port
        self._n_gpu_layers = n_gpu_layers
        self._context_size = context_size
        self._startup_timeout_sec = startup_timeout_sec

        # Estado del subproceso y modelo activo
        self._proc: Optional[subprocess.Popen] = None
        self._active_model_id: Optional[str] = None
        self._active_model_path: Optional[str] = None
        self._start_time: float = 0.0
        self._generation_count: int = 0
        self._last_tok_s: float = 0.0
        self._last_ttft_ms: float = 0.0
        self._cancel_events: Dict[str, asyncio.Event] = {}

    # ── Capacidades ─────────────────────────────────────────────

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_gpu=True,
            supports_npu=False,
            supports_streaming=True,
            supports_speculative_decoding=False,
            supports_multi_model=False,
            supports_vision=False,
            supports_audio=False,
            max_context_length=self._context_size,
            supported_quantizations=("int4", "q4_k_m", "q5_k_m", "q8_0", "f16"),
            provider_type=ProviderType.LLAMACPP,
        )

    # ── Ciclo de Vida ───────────────────────────────────────────

    async def initialize(self) -> None:
        """Valida que el binario de llama-server.exe exista y sea ejecutable."""
        if self._status == ProviderStatus.READY:
            return

        self._status = ProviderStatus.INITIALIZING

        # Verificar existencia del ejecutable
        if not os.path.exists(self._server_bin_path):
            alt_path = shutil.which("llama-server.exe") or shutil.which("llama-server")
            if alt_path and os.path.exists(alt_path):
                self._server_bin_path = alt_path
            else:
                self._status = ProviderStatus.ERROR
                self._last_error = f"Binario no encontrado en: {self._server_bin_path}"
                logger.error(f"[LlamaCppProvider] {self._last_error}")
                return

        self._status = ProviderStatus.READY
        logger.info(f"[LlamaCppProvider] Inicializado con binario: {self._server_bin_path}")

    async def shutdown(self) -> None:
        """Detiene el servidor y libera todos los recursos del sistema."""
        self._status = ProviderStatus.SHUTTING_DOWN
        if self._active_model_id:
            await self.unload_model(self._active_model_id)
        self._status = ProviderStatus.SHUTDOWN
        logger.info("[LlamaCppProvider] Servidor apagado y recursos liberados.")

    # ── Gestión de Modelos ──────────────────────────────────────

    async def load_model(self, model_id: str, model_path: str) -> None:
        """Inicia llama-server.exe con el modelo GGUF y espera su disponibilidad."""
        if self._active_model_id == model_id and await self.health_check():
            logger.info(f"[LlamaCppProvider] Modelo '{model_id}' ya está cargado y listo.")
            return

        # Si hay otro proceso en ejecución, detenerlo primero
        if self._proc is not None:
            await self.unload_model(self._active_model_id or "unknown")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Archivo de modelo GGUF no encontrado: {model_path}")

        # Comprobar colisión de puertos y asignar puerto libre si es necesario
        self._port = _find_available_port(self._host, self._port)

        cmd = [
            self._server_bin_path,
            "-m", model_path,
            "-ngl", str(self._n_gpu_layers),
            "-c", str(self._context_size),
            "--host", self._host,
            "--port", str(self._port),
        ]

        logger.info(f"[LlamaCppProvider] Lanzando daemon: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # Iniciar subproceso desacoplado
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        self._start_time = time.time()
        self._status = ProviderStatus.INITIALIZING

        # Polling de readiness en endpoint /health
        ready = await self._wait_for_server_ready()
        if not ready:
            await self.unload_model(model_id)
            raise TimeoutError(f"llama-server no respondió en el tiempo límite ({self._startup_timeout_sec}s)")

        self._active_model_id = model_id
        self._active_model_path = model_path
        self._status = ProviderStatus.READY
        logger.info(f"[LlamaCppProvider] Servidor listo para inferencia (PID {self._proc.pid}, puerto {self._port})")

    async def _wait_for_server_ready(self) -> bool:
        """Consulta periódicamente /health hasta que el servidor devuelva status ok."""
        t_end = time.time() + self._startup_timeout_sec
        url = f"http://{self._host}:{self._port}/health"

        while time.time() < t_end:
            if self._proc is not None and self._proc.poll() is not None:
                logger.error(f"[LlamaCppProvider] El subproceso terminó inesperadamente (código {self._proc.returncode})")
                return False

            try:
                loop = asyncio.get_running_loop()
                is_ok = await loop.run_in_executor(None, self._sync_health_check, url)
                if is_ok:
                    return True
            except Exception:
                pass

            await asyncio.sleep(0.5)

        return False

    def _sync_health_check(self, url: str) -> bool:
        """Comprobación síncrona HTTP de /health."""
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("status") in ("ok", "ready") or "ok" in str(data).lower()
        except Exception:
            return False
        return False

    async def unload_model(self, model_id: str) -> None:
        """Detiene de forma limpia el proceso de llama-server y libera VRAM."""
        if self._proc is None:
            self._active_model_id = None
            self._active_model_path = None
            return

        logger.info(f"[LlamaCppProvider] Deteniendo llama-server (PID {self._proc.pid})...")
        try:
            self._proc.terminate()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._wait_process_exit)
        except Exception as e:
            logger.warning(f"[LlamaCppProvider] Error al terminar proceso: {e}")
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None
            self._active_model_id = None
            self._active_model_path = None
            self._status = ProviderStatus.READY
            # Breve pausa para asegurar liberación de VRAM en driver
            await asyncio.sleep(0.5)
            logger.info("[LlamaCppProvider] Proceso detenido con éxito.")

    def _wait_process_exit(self) -> None:
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2.0)

    async def is_model_loaded(self, model_id: str) -> bool:
        return (
            self._active_model_id == model_id
            and self._proc is not None
            and self._proc.poll() is None
            and await self.health_check()
        )

    async def loaded_models(self) -> List[str]:
        return [self._active_model_id] if self._active_model_id else []

    # ── Inferencia ──────────────────────────────────────────────

    def _build_openai_payload(self, request: InferenceRequest, stream: bool) -> Dict[str, Any]:
        """Construye el payload compatible con OpenAI /v1/chat/completions."""
        messages: List[Dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: Dict[str, Any] = {
            "model": request.model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        return payload

    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Inferencia atómica síncrona vía /v1/chat/completions."""
        if not await self.health_check():
            raise RuntimeError("llama-server no está disponible para inferencia.")

        self._status = ProviderStatus.BUSY
        t0 = time.perf_counter()
        url = f"http://{self._host}:{self._port}/v1/chat/completions"
        payload = self._build_openai_payload(request, stream=False)

        try:
            loop = asyncio.get_running_loop()
            resp_data = await loop.run_in_executor(None, self._http_post_json, url, payload)
            elapsed_sec = time.perf_counter() - t0

            content = resp_data["choices"][0]["message"]["content"]
            usage = resp_data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", len(content.split()))
            prompt_tokens = usage.get("prompt_tokens", 0)

            tok_s = completion_tokens / elapsed_sec if elapsed_sec > 0 else 0.0
            self._last_tok_s = tok_s
            self._generation_count += 1

            return InferenceResult(
                text=content,
                finish_reason=resp_data["choices"][0].get("finish_reason", "stop"),
                tokens_generated=completion_tokens,
                prompt_tokens=prompt_tokens,
                latency_ms=elapsed_sec * 1000.0,
                tokens_per_sec=tok_s,
                model_id=request.model_id,
                provider_type=ProviderType.LLAMACPP.value,
            )
        finally:
            self._status = ProviderStatus.READY

    def _http_post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Envía POST JSON usando urllib estándar de Python."""
        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def generate_stream(self, request: InferenceRequest) -> AsyncIterator[InferenceResult]:
        """Inferencia token-a-token reactiva mediante SSE (Server-Sent Events)."""
        if not await self.health_check():
            raise RuntimeError("llama-server no está disponible para inferencia streaming.")

        self._status = ProviderStatus.BUSY
        cancel_event = asyncio.Event()
        if request.request_id:
            self._cancel_events[request.request_id] = cancel_event

        t_start = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens_generated = 0
        url = f"http://{self._host}:{self._port}/v1/chat/completions"
        payload = self._build_openai_payload(request, stream=True)

        try:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

            # Tarea en hilo para consumir el stream HTTP bloqueante
            def _stream_worker():
                try:
                    body_bytes = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=body_bytes,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=120.0) as resp:
                        for line in resp:
                            if cancel_event.is_set():
                                break
                            line_str = line.decode("utf-8", errors="replace").strip()
                            if not line_str or line_str.startswith(":"):
                                continue
                            if line_str.startswith("data: "):
                                data_part = line_str[6:].strip()
                                if data_part == "[DONE]":
                                    break
                                loop.call_soon_threadsafe(queue.put_nowait, data_part)
                except Exception as ex:
                    logger.debug(f"[LlamaCppProvider] Stream worker finalizado o interrumpido: {ex}")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            worker_future = loop.run_in_executor(None, _stream_worker)

            while True:
                if cancel_event.is_set():
                    yield InferenceResult(
                        text="",
                        finish_reason="stop",
                        tokens_generated=tokens_generated,
                        model_id=request.model_id,
                        provider_type=ProviderType.LLAMACPP.value,
                    )
                    return

                item = await queue.get()
                if item is None:
                    break

                try:
                    chunk_json = json.loads(item)
                    choices = chunk_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        token_text = delta.get("content", "")
                        finish_reason = choices[0].get("finish_reason")

                        if token_text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                self._last_ttft_ms = (first_token_time - t_start) * 1000.0

                            tokens_generated += 1
                            yield InferenceResult(
                                text=token_text,
                                finish_reason=finish_reason,
                                tokens_generated=tokens_generated,
                                model_id=request.model_id,
                                provider_type=ProviderType.LLAMACPP.value,
                            )
                except Exception:
                    continue

            elapsed = time.perf_counter() - t_start
            if elapsed > 0 and tokens_generated > 0:
                self._last_tok_s = tokens_generated / elapsed
            self._generation_count += 1

        finally:
            self._status = ProviderStatus.READY
            if request.request_id in self._cancel_events:
                del self._cancel_events[request.request_id]

    async def cancel_generation(self, request_id: str) -> None:
        """Cancela un stream en progreso."""
        if request_id in self._cancel_events:
            self._cancel_events[request_id].set()
            logger.info(f"[LlamaCppProvider] Cancelación solicitada para request_id={request_id}")

    # ── Telemetría y Salud ──────────────────────────────────────

    async def health_check(self) -> bool:
        """Comprueba de forma no bloqueante si llama-server está saludable."""
        if self._proc is None or self._proc.poll() is not None:
            return False
        url = f"http://{self._host}:{self._port}/health"
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._sync_health_check, url)
        except Exception:
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """Devuelve el estado operativo y métricas de inferencia del servidor."""
        uptime = (time.time() - self._start_time) if self._start_time > 0 else 0.0
        return {
            "backend": "llamacpp",
            "model_id": self._active_model_id,
            "process_pid": self._proc.pid if self._proc else None,
            "server_port": self._port,
            "loaded": self._proc is not None and self._proc.poll() is None,
            "uptime_seconds": round(uptime, 2),
            "last_error": self._last_error,
            "generation_count": self._generation_count,
            "last_tok_s": round(self._last_tok_s, 2),
            "last_ttft_ms": round(self._last_ttft_ms, 2),
        }
