# MULTI_BACKEND_P0_AUDIT.md
# AS-Core — Auditoría Técnica de Interfaces y Puntos de Integración (Fase P0)
### Estado: **AUDITORÍA TÉCNICA COMPLETADA (CERO CÓDIGO MODIFICADO)**
### Fecha: 2026-08-27 | Nivel de Revisión: **GEMINI 3.7 HIGH & MEDIUM**

---

## 1. Resumen Ejecutivo de la Auditoría P0

La auditoría de código del repositorio AS-Code confirma que **la arquitectura de AS-Core ya cuenta con un desacoplamiento limpio entre la capa cognitiva superior y los motores de inferencia**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ARQUITECTURA DE INTEGRACIÓN ENCONTRADA                          │
├────────────────────────────────┬───────────────────────────────────────────────────────┤
│ Capa Cognitiva                 │ `Coordinator` / `AgentControlRunner` / `StateStore`   │
│                                │ • Cero dependencia de librerías de inferencia.        │
│                                │ • Interactúa exclusivamente con `EngineManager`.      │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Capa de Orquestación           │ `core.engine.EngineManager`                           │
│                                │ • Administra ciclo de vida, anti-OOM y hot-swap.      │
│                                │ • Delega llamadas a `InferenceProvider.generate()` y  │
│                                │   `generate_stream()`.                                │
├────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Capa de Abstracción de Motor   │ `providers.base.InferenceProvider` & `ProviderRegistry│
│                                │ • Contrato abstracto asíncrono.                       │
│                                │ • Soporta múltiples proveedores registrados.         │
└────────────────────────────────┴───────────────────────────────────────────────────────┘
```

**Conclusión Fundamental de P0:**  
Para integrar `llama.cpp` y soportar `Qwen1.5-MoE-A2.7B` **NO se requiere refactorizar el Coordinator, Skills, RAG, Memory ni LiteRT**. Únicamente se necesita crear un nuevo proveedor aislado (`LlamaCppProvider`) que implemente el contrato existente `InferenceProvider`.

---

## 2. Respuestas Específicas a las 10 Preguntas Críticas de P0

### 1. ¿Cómo se registra actualmente un proveedor?
En `providers/registry.py`:
```python
registry = ProviderRegistry()
registry.register("litert_embedded", LiteRTEmbeddedProvider(...))
registry.register("litert_cli", LiteRTCLIProvider(...))
```
El registro almacena la instancia en `_providers[provider_id]` sin inicializarla de inmediato (construcción ligera).

### 2. ¿Cómo `EngineManager` selecciona y carga un modelo?
En `core/engine.py`:
- Los modelos se registran mediante `register_model(model_id, path, type, estimated_vram_mb, provider_id)`.
- Al recibir una solicitud `generate()` o `generate_stream()`, invoca `_ensure_model_loaded(model_id)`:
  1. Obtiene `provider = registry.get_provider(provider_id)`.
  2. Verifica `is_model_loaded(model_id)`. Si ya está cargado, retorna inmediatamente.
  3. Ejecuta `_check_resources(estimated_vram_mb)`.
  4. Si hay otro modelo cargado en VRAM, llama a `provider.unload_model(old_model_id)` para liberar la memoria.
  5. Llama a `await provider.load_model(model_id, path)` y actualiza `_last_used` y `_loaded_at`.

### 3. ¿Qué contrato real espera actualmente `InferenceProvider`?
Definido en `providers/base.py` (`class InferenceProvider(ABC)`):
- **Lifecycle:** `initialize()`, `shutdown()`.
- **Capacidades:** `capabilities() -> ProviderCapabilities`.
- **Modelos:** `load_model(model_id, model_path)`, `unload_model(model_id)`, `is_model_loaded(model_id) -> bool`, `loaded_models() -> list[str]`.
- **Inferencia:** `generate(request: InferenceRequest) -> InferenceResult`, `generate_stream(request: InferenceRequest) -> AsyncIterator[InferenceResult]`, `cancel_generation(request_id: str)`.
- **Telemetría:** `health_check() -> bool`, `get_metrics() -> dict`.

### 4. ¿Qué modificaciones mínimas son necesarias para soportar `LlamaCppProvider`?
- **En `providers/base.py`:** Añadir `LLAMACPP = "llamacpp"` al enum `ProviderType`. (1 línea de cambio).
- **Nuevo archivo `providers/llamacpp_provider.py`:** Implementar `LlamaCppProvider(InferenceProvider)` que gestione el ciclo de vida de `llama-server.exe` y consuma su API HTTP/SSE OpenAI-compatible.
- **En `config.yaml`:** Añadir la entrada del modelo MoE y la sección `providers.llamacpp`.

### 5. ¿Cómo iniciar y detener `llama-server.exe` de forma segura en Windows?
- **Inicio:**
  - Comprobar disponibilidad de puerto (default `8766`) mediante `socket.socket()`.
  - Lanzar `subprocess.Popen` con flags de desvinculación o variables de entorno `PYTHONUNBUFFERED=1`, `PYTHONIOENCODING=utf-8`, capturando `stdout=subprocess.PIPE` y `stderr=subprocess.PIPE`.
  - Guardar el `pid` del proceso en `self._proc`.
- **Detención Garantizada:**
  - Enviar `SIGTERM` / `proc.terminate()`.
  - Esperar hasta `5.0 s` con `proc.wait()`.
  - Si el proceso no termina, forzar `proc.kill()` y asegurar cierre de sockets.
  - Comprobar liberación de memoria VRAM (`nvidia-smi` / `cuda_driver`).

### 6. ¿Cómo comprobar `/health` antes de declarar el backend disponible?
- En `initialize()` o `load_model()`:
  - Realizar un bucle asíncrono `GET http://127.0.0.1:8766/health` con `asyncio.sleep(0.5)` hasta timeout de 60s.
  - Cuando la respuesta HTTP sea `200 OK` con `{"status": "ok"}` (o `loading model` pase a `ready`), cambiar `self._status = ProviderStatus.READY`.

### 7. ¿Cómo consumir `/v1/chat/completions` con streaming SSE?
- En `generate_stream(request: InferenceRequest)`:
  - Enviar `POST http://127.0.0.1:8766/v1/chat/completions` con `stream: True` usando `httpx.AsyncClient` o cliente SSE streaming asíncrono.
  - Parsear líneas `data: {"choices": [{"delta": {"content": "..."}}]}`.
  - Emitir incrementalmente `InferenceResult(text=token, model_id=request.model_id, provider_type="llamacpp")`.
  - Manejar `cancel_event`: al cancelarse, abortar el stream HTTP inmediatamente para que `llama-server` libere el slot de inferencia.

### 8. ¿Cómo evitar conflictos de VRAM entre Gemma, OLMoE y llama.cpp?
- La política de `EngineManager` ya implementa el principio de **Single-Active-Model** para GPUs de nivel `BALANCED` / `ULTRA_LIGHT` (como la GTX 1650 Ti 4GB):
  - Antes de cargar `qwen-moe` en `llama.cpp` (~3.9 GB VRAM), `EngineManager` invoca `unload_model()` sobre `LiteRT` o `ASMoE`.
  - Antes de cargar `gemma` en `LiteRT` (~1.5 GB VRAM), `EngineManager` invoca `unload_model()` sobre `LlamaCppProvider`, terminando el subproceso `llama-server.exe` y dejando libre la VRAM.

### 9. ¿Cómo permitir que AS-Core cambie entre los tres modelos sin modificar Coordinator, Skills, RAG ni Memory?
- `Coordinator` y `AgentControlRunner` solicitan inferencia enviando un `InferenceRequest(model_id=...)` a `EngineManager`.
- `EngineManager` busca el modelo en `_model_configs[model_id]` y resuelve automáticamente el `provider_id` correspondiente (`litert_embedded`, `llamacpp`, o `as_moe`).
- **El flujo superior permanece 100% idéntico e inmutable.**

### 10. ¿Qué archivos concretos deberán modificarse en P1?
1. **`providers/base.py`:** Agregar `LLAMACPP = "llamacpp"` a `ProviderType`.
2. **`providers/llamacpp_provider.py`:** `[NUEVO]` Implementación completa del proveedor para Windows CUDA.
3. **`providers/__init__.py`:** Exportar `LlamaCppProvider`.
4. **`config.yaml`:** Añadir configuración del modelo `qwen-moe` y backend `llamacpp`.
5. **`tests/test_llamacpp_provider.py`:** `[NUEVO]` Tests unitarios aislados del ciclo de vida y streaming.

---

## 3. Matriz de Archivos Intocables vs Modificables

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MATRIZ DE AISLAMIENTO P1                                  │
├─────────────────────────────────────────────────────────┬──────────────────────────────┤
│ Archivo / Directorio                                    │ Estado / Directiva           │
├─────────────────────────────────────────────────────────┼──────────────────────────────┤
│ `runtime/coordinator/*` (Agent, Manager, StateStore)    │ 🔒 PROHIBIDO TOCAR (Intacto) │
│ `runtime/skills/*` (Governance, Registry, Sandbox)      │ 🔒 PROHIBIDO TOCAR (Intacto) │
│ `runtime/memory/*` (Conversations, WorkingMemory)       │ 🔒 PROHIBIDO TOCAR (Intacto) │
│ `core/moe/*` (Registry, Pools, Residency Engine)        │ 🔒 PROHIBIDO TOCAR (Intacto) │
│ `providers/litert_embedded.py` / `litert_cli.py`        │ 🔒 PROHIBIDO TOCAR (Intacto) │
│ `api/*` (Endpoints, Server, Routes)                     │ 🔒 PROHIBIDO TOCAR (Intacto) │
├─────────────────────────────────────────────────────────┼──────────────────────────────┤
│ `providers/base.py`                                     │ ✏️ Añadir 1 línea (Enum)     │
│ `providers/llamacpp_provider.py`                        │ ➕ ARCHIVO NUEVO             │
│ `config.yaml`                                           │ ✏️ Añadir entrada de modelo  │
│ `tests/test_llamacpp_provider.py`                       │ ➕ ARCHIVO NUEVO             │
└─────────────────────────────────────────────────────────┴──────────────────────────────┘
```

---

## 4. Análisis de Riesgos y Estrategia de Rollback

| Riesgo | Probabilidad | Impacto | Mitigación Técnica en P1 |
|---|---|---|---|
| **Colisión de Puertos (8766)** | Media | Medio | Detección previa de socket libre; selector dinámico de puerto si está ocupado. |
| **Procesos Zombi en Windows** | Media | Alto | Limpieza explícita en `shutdown()`, uso de `atexit` y verificación de PID huérfano. |
| **Conflicto de VRAM / OOM** | Baja | Alto | Garantizar `unload_model()` previo antes de inicializar el nuevo subproceso. |
| **Timeout en Carga Inicial** | Baja | Medio | Timeout holgado de 120s en polling de `/health` con logging progresivo. |

### Estrategia de Rollback Inmediato:
Si `LlamaCppProvider` experimenta fallos durante `load_model()` o `health_check()`, `EngineManager` captura la excepción, mantiene intacto el proveedor activo previo (`litert_embedded` / Gemma) y emite un mensaje de error controlado sin degradar el runtime de AS-Core.

---

## 5. Batería de Tests Previa a P1 (`tests/test_llamacpp_provider.py`)

Para considerar cumplido el Gate de P1 se requerirán los siguientes tests unitarios:
1. `test_llamacpp_capabilities_and_initialization`: Verificación de binario y capacidades reportadas.
2. `test_llamacpp_process_lifecycle_and_health`: Arranque de `llama-server.exe`, validación de `/health` y apagado limpio.
3. `test_llamacpp_atomic_generate`: Inferencia no-streaming con `qwen1.5-moe-a2.7b`.
4. `test_llamacpp_streaming_generation`: Streaming token-a-token SSE sin pérdida de chunks.
5. `test_llamacpp_cancellation`: Interrupción limpia de stream a mitad de generación.
6. `test_llamacpp_vram_cleanup_on_shutdown`: Verificación de que la VRAM de la GPU se libera al 100%.

---

## 6. Dictamen y Estado del Gate P0

```
================================================================================
 DICTAMEN DEL GATE P0:
================================================================================
 [x] Código real de AS-Core auditado exhaustivamente.
 [x] Cero modificaciones a componentes existentes realizadas.
 [x] Contrato de InferenceProvider validado y 100% compatible.
 [x] Plan de integración no destructivo formalizado.
 [x] Matriz de archivos intocables y estrategia de rollback definidas.

 ESTADO: GREEN ✅
 ESPERANDO AUTORIZACIÓN PARA PROCEDER CON: P1 — LlamaCppProvider mínimo y aislado
================================================================================
```
