# MULTI_BACKEND_IMPLEMENTATION_PLAN.md
# AS-Core — Plan Maestro de Arquitectura Multi-Backend & Roadmap MoE
### Estado: **DOCUMENTO DE PLANIFICACIÓN FORMAL (PRE-IMPLEMENTACIÓN)**
### Fecha: 2026-08-27 | Clasificación: **GEMINI 3.7 HIGH & MEDIUM**

---

## 1. Declaración de Filosofía y Principio Arquitectónico

> **DIRECTIVA FUNDAMENTAL DE AS-CORE:**  
> **AS-Core NO debe depender de un único motor de inferencia ni acoplar su inteligencia cognitiva a un backend específico.**

El subsistema cognitivo superior de AS-Core (**Coordinator, Skills, RAG, Memory, Working Memory, Projects, Chats, Context Resolution**) debe operar a través de un contrato abstracto e inmutable (`InferenceBackend`), siendo completamente agnóstico a si la inferencia se ejecuta mediante **LiteRT-LM**, **llama.cpp CUDA nativo**, o nuestro motor experimental **AS-Core MoE Engine**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               AS-CORE COGNITIVE RUNTIME                                │
│       (Coordinator, Agent, Skills, RAG, Memory, Projects, StateStore, Streaming)       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │     InferenceBackend    │
                               │  (Contrato Unificado)   │
                               └────────────┬────────────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         │                                  │                                  │
┌────────▼────────┐                ┌────────▼────────┐                ┌────────▼────────┐
│  LiteRTBackend  │                │  LlamaCppBackend│                │   MoEBackend    │
├─────────────────┤                ├─────────────────┤                ├─────────────────┤
│ Modelo: Gemma   │                │ Modelo: Qwen    │                │ Modelo: OLMoE   │
│ E2B / E4B       │                │ MoE A2.7B       │                │ 1B-7B / Custom  │
│ Backend: LiteRT │                │ Backend:        │                │ Backend:        │
│ C++ / DirectX   │                │ llama.cpp CUDA  │                │ AS-Core Dynamic │
│ Rol: Fast / Low │                │ Rol: MoE Grande │                │ Residency Engine│
│ Latency Baseline│                │ Referencia      │                │ Rol: I+D Lab    │
│ [VERIFIED]      │                │ [MEASURED: 18.6]│                │ [MEASURED: 65.4]│
└─────────────────┘                └─────────────────┘                └─────────────────┘
```

---

## 2. Auditoría del Estado Actual de AS-Core

### 2.1. Componentes Cognitivos y Estructura Existente [VERIFIED]
- **`providers/base.py` & `providers/registry.py`:**
  - Existe una abstracción base `InferenceProvider` con `InferenceRequest`, `InferenceResult`, `ProviderCapabilities` y `ProviderRegistry`.
  - Actualmente orientada primordialmente a tipos `LITERT_CLI`, `LITERT_COMPILED`, `LITERT_NATIVE`.
  - **Diagnóstico:** La estructura de `ProviderRegistry` y `EngineManager` es sólida y extensible; requiere evolucionar hacia `ProviderType.LLAMACPP` y `ProviderType.AS_CORE_MOE`.
- **`core/engine.py` (`EngineManager`):**
  - Orquesta el ciclo de vida del modelo, presupuestos de VRAM y políticas de descarga por inactividad.
  - Se comunica exclusivamente con el proveedor activo mediante `generate()` y `generate_stream()`.
- **`runtime/coordinator/` (Agent, Coordinator, Intent, StateStore, Workflow):**
  - Diseñado de forma modular. Consume la inferencia vía `EngineManager` o APIs OpenAI-compatibles.
  - **Diagnóstico:** Cero dependencias directas de LiteRT en la capa de razonamiento. La integración multi-backend es **100% no destructiva**.
- **`core/moe/` (B1 $\to$ B4.4):**
  - `DynamicResidencyEngine`, `RealRouter`, `ExpertRegistry`, `VRAMExpertPool`, `RAMWarmPool`, `MoELayerExecutor`, `RoutingTracer`, `FrequencyAnalyzer`.
  - 57/57 tests en verde (`100% PASS`).
  - Capacidad de ejecutar capas MoE completas in-GPU y perfiles de residencia física.

---

## 3. Arquitectura Propuesta: Abstracción `InferenceBackend`

### 3.1. Contrato Unificado de Backend (`providers/base.py`)
Todo backend de inferencia debe implementar estrictamente la siguiente interfaz asíncrona:

```python
class InferenceBackend(ABC):
    """Contrato agnóstico obligatorio para todos los motores de inferencia en AS-Core."""

    @abstractmethod
    async def initialize(self) -> None:
        """Inicializa buffers de hardware, GPU contexts o subprocesos daemon."""
        pass

    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResult:
        """Inferencia atómica no bloqueante."""
        pass

    @abstractmethod
    def stream(self, request: InferenceRequest) -> AsyncIterator[InferenceResult]:
        """Generación token-a-token con streaming reactivo."""
        pass

    @abstractmethod
    async def health(self) -> Dict[str, Any]:
        """Estado físico del motor, VRAM libre, temperatura y disponibilidad."""
        pass

    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        """Metadatos del modelo cargado (arquitectura, parámetros, cuantización)."""
        pass

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Capacidades del motor (streaming, GPU, MoE, contexto máximo)."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Liberación garantizada de VRAM y descriptores de proceso."""
        pass
```

### 3.2. Catálogo de Backends Oficiales
1. **`LiteRTBackend` (`providers/litert_embedded.py` / `litert_cli.py`):**
   - Maneja modelos densos Gemma 2B / Gemma 4B.
   - Enlace optimizado para Windows GPU/DirectX.
2. **`LlamaCppBackend` (`providers/llamacpp_provider.py` - NUEVO):**
   - Maneja modelos MoE grandes (`Qwen1.5-MoE-A2.7B-Chat`, `DeepSeek-V2-Lite`).
   - Ejecuta `llama-server.exe` en modo subproceso daemon con API HTTP/SSE OpenAI-compatible o DLL nativa C/CUDA (`llama.dll`).
3. **`MoEBackend` / `OLMoEBackend` (`providers/as_moe_provider.py` - NUEVO):**
   - Maneja modelos MoE de investigación (`OLMoE-1B-7B`, experimentación de residency).
   - Conecta con `core.moe.dynamic_residency_engine`.

---

## 4. Integración Detallada de los 3 Motores

### 4.1. Backend 1: Gemma E2B (LiteRT-LM)
- **Rol:** Inferencia ultra-rápida de baja latencia para tareas cognitivas inmediatas, clasificación de intenciones y generación concisa.
- **Modelo:** `gemma-2b-it-gpu-int4.bin` / `gemma-e2b`.
- **Estado:** `[VERIFIED]` Estable en AS-Core. **Regla de oro: Cero cambios disruptivos.**

### 4.2. Backend 2: OLMoE (AS-Core MoE Engine)
- **Rol:** Laboratorio de investigación GPU-First para arquitecturas MoE que caben 100% en VRAM o requieren segmentación controlada.
- **Modelo:** `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` (3.92 GB en VRAM).
- **Rendimiento Medido [MEASURED]:** **`65.37 tok/s`** en NVIDIA GTX 1650 Ti 4GB GDDR6.
- **Integración:** Encapsulado mediante `MoEBackend` conectando `ModelProfile` y `ExpertRegistry`.

### 4.3. Backend 3: Modelo MoE Grande (`llama.cpp` Windows CUDA Nativo)
- **Rol:** Backend de producción de referencia para modelos MoE que exceden la memoria VRAM disponible (ej: 8.84 GB en 4 GB VRAM).
- **Modelo:** `Qwen1.5-MoE-A2.7B-Chat-Q4_K_M.gguf` (14.3B / 2.7B activos).
- **Rendimiento Medido [MEASURED]:** **`18.64 tok/s`** (Modo Híbrido `-ngl 10`).
- **Análisis de Capacidades Relevantes de llama.cpp:**
  - `llama-server.exe` build 10649 con `ggml-cuda.dll` (CUDA 12.4 Turing sm_75).
  - **Offload por Capas (`-ngl`):** Permite colocar $N$ capas completas en GPU y el resto en CPU RAM.
  - **CPU MoE (`-ncmoe 1`):** Mantiene la atención densa en GPU y deriva únicamente los expertos MoE a CPU RAM (**`10.51 tok/s`** `[MEASURED]`).
  - **KV Cache Offload (`-fa` / `cache_type_k/v`):** Aceleración de atención en VRAM.
  - **API OpenAI Compatible:** Endpoints `/v1/chat/completions`, `/v1/completions`, `/health`, `/slots` con soporte nativo de SSE streaming (`text/event-stream`).

---

## 5. Análisis Comparativo: llama.cpp vs AS-Core Residency Engine

### 5.1. Comparativa de Rendimiento Empírico en Hardware Idéntico [MEASURED]

| Métrica | llama.cpp Híbrido (`-ngl 10`) | AS-Core Residency Engine (12 slots) | Diferencia / Factor |
|---|---|---|---|
| **Velocidad de Generación** | **`18.64 tok/s`** | **`4.98 tok/s`** | **llama.cpp es 3.74x más rápido** |
| **Latencia al Primer Token (TTFT)** | $480.0\text{ ms}$ | **`392.6 ms`** | **AS-Core es 1.22x más rápido (22% mejor)** |
| **Hit Rate de Residencia VRAM** | $41.7\%$ (10 capas fijas) | **`72.11%` (LRU dinámico)** | **AS-Core retiene +30.4% más hits** |
| **VRAM Utilizada** | $3,893\text{ MB}$ ($95.0\%$) | **`1,732.5 MB` ($42.3\%$)** | **AS-Core ahorra 2,160 MB de VRAM** |
| **VRAM Headroom Libre** | $+203\text{ MB}$ (Crítico) | **$+2,363\text{ MB}$ (Holgado)** | AS-Core permite contextos mayores |

### 5.2. Diagnóstico Técnico: Hipótesis de Rendimiento y Desglose a Medir

> [!NOTE]
> **Aclaración Epistemológica:** La diferencia entre $18.64\text{ tok/s}$ (llama.cpp) y $4.98\text{ tok/s}$ (prototipo Python AS-Core) NO debe atribuirse de forma simplista o definitiva a "Python overhead" de forma exclusiva. Existen indicios técnicos muy fuertes (ej: slicing en CPU de memmap en cada fallo de página), pero **debe medirse de forma aislada y rigurosa cada etapa del pipeline** para determinar exactamente la distribución de la latencia.

El plan de perfilado detallado descompondrá el ciclo de inferencia en:
1. **Router Forward Pass:** Latencia de proyección logits y Softmax Top-K (`RealRouter`).
2. **Expert Lookup & LRU Management:** Resolución O(1) de slots en RAM/VRAM.
3. **Transferencia RAM $\to$ VRAM:** Medición directa del ancho de banda PCIe (Pinned DMA vs Pageable).
4. **CUDA Launch Overhead:** Coste de invocación de kernels desde el host.
5. **Cómputo SwiGLU Expert GEMM:** Rendimiento bruto de los GEMMs de gate, up y down en GPU.
6. **Weighted Sum Accumulation:** Tiempo de acumulación in-GPU con `cublasSaxpy`.
7. **Attention & RoPE:** Latencia de capas de atención densa en GPU.
8. **KV Cache Management:** Acceso y actualización de memoria de contexto.
9. **CPU Orchestration & Runtime:** Slicing de tensores, gestión de memoria y overhead del runtime / GIL.
10. **Latencia Total de Token:** Suma end-to-end vs tok/s sostenidos.

---

## 6. Roadmap de Optimización de Nuestro MoE Residency Engine

Para que el **AS-Core MoE Engine** sea competitivo frente a `llama.cpp` en modelos grandes (>4 GB), no basta con ajustar parámetros; debe eliminarse el overhead de orquestación en CPU:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           ROADMAP DE OPTIMIZACIÓN MOE ENGINE                           │
├─────────┬──────────────────────────────────┬─────────────────────────────┬─────────────┤
│ Nivel   │ Optimización Propuesta           │ Impacto Proyectado          │ Tok/s Est.  │
├─────────┼──────────────────────────────────┼─────────────────────────────┼─────────────┤
│ **O1**  │ Pre-ensamblado en `RAMWarmPool`  │ Elimina slicing NumPy (0 ms)│ 9.5 tok/s   │
│ **O2**  │ Asynchronous CUDA Streams (Async)│ Oculta PCIe tras cómputo    │ 12.8 tok/s  │
│ **O3**  │ Pre-fetching del siguiente token │ Anticipa carga de expertos  │ 15.5 tok/s  │
│ **O4**  │ Kernel C++/CUDA unificado (DLL)  │ Elimina GIL y launch delays │ **`>20 tok/s`**│
└─────────┴──────────────────────────────────┴─────────────────────────────┴─────────────┘
```

---

## 7. Plan de Benchmark Común y Batería de Pruebas de Calidad

### 7.1. Matriz de Benchmark Homogéneo
Se evaluarán los 3 backends bajo condiciones idénticas:
- **Hardware:** GTX 1650 Ti 4GB / Intel Core i5-10300H / 16GB RAM / NVMe / Windows 11.
- **Parámetros de Inferencia:** `temperature = 0.0`, `max_tokens = 128`, `top_p = 1.0`.
- **Métricas:** TTFT (ms), tok/s, uso de VRAM (MB), uso de RAM (GB), % CPU, % GPU, bytes PCIe.

### 7.2. Batería de Pruebas de Calidad Cognitiva (8 Dimensiones)
1. **Conversación:** Fluidez, coherencia, tono y seguimiento de instrucciones en español e inglés.
2. **Razonamiento:** Resolución lógica de problemas y derivaciones paso a paso.
3. **Generación de Código:** Sintaxis exacta en Python, manejo de estructuras de datos y algoritmos.
4. **Recuperación Documental (RAG):** Precisión en la síntesis de contexto provisto y ausencia de alucinaciones.
5. **Memoria de Conversación:** Retención de entidades y acuerdos de turnos anteriores.
6. **Ejecución de Skills:** Selección correcta de herramientas y llamadas de función estructuradas.
7. **Contexto Largo (1K a 2K tokens):** Coherencia semántica y consistencia temporal en respuestas extensas.
8. **Instrucciones Complejas:** Cumplimiento de restricciones múltiples y formatos estructurados (JSON/Markdown).

---

## 8. Análisis FODA (SWOT) de la Estrategia Multi-Backend

### Fortalezas (Strengths)
- **Independencia Tecnológica:** AS-Core no queda atrapado en las limitaciones de ningún motor individual.
- **Aprovechamiento Óptimo de Hardware:** Selecciona el motor ideal según el tamaño del modelo (Gemma para ultra-rápido, OLMoE para GPU total, llama.cpp para MoE grande).
- **Rendimiento de Producción Inmediato:** Garantiza $\ge 18.6\text{ tok/s}$ hoy mediante `llama.cpp` sin comprometer la estabilidad.

### Oportunidades (Opportunities)
- **Evolución Modular:** Permite actualizar `llama.cpp` de forma binaria sin tocar el código cognitivo de AS-Core.
- **I+D Diferenciada:** Nuestro MoE Engine puede concentrarse en algoritmos avanzados de predicción de expertos y hotsets dinámicos.

### Debilidades (Weaknesses)
- **Mantenimiento de Múltiples Adaptadores:** Requiere mantener wrappers limpios para LiteRT, llama.cpp y MoE Engine.
- **Proceso Externo para llama.cpp:** Requiere gestionar el ciclo de vida del subproceso `llama-server.exe` en Windows.

### Amenazas (Threats)
- **Cambios en APIs Upstream:** Actualizaciones de versiones de llama.cpp o LiteRT que modifiquen flags de CLI o endpoints.
- **Presupuesto Estricto de VRAM:** Riesgo de contención de memoria si dos motores intentan coexistir en VRAM simultáneamente.

---

## 9. Criterios de Éxito, Gates y Criterios de Abandono

### 9.1. Gates Obligatorios
- **GATE A:** Gemma E2B continúa funcionando al 100% sin regresiones en LiteRT-LM.
- **GATE B:** OLMoE continúa funcionando y mantiene su baseline de $\ge 60\text{ tok/s}$ en VRAM.
- **GATE C:** `LlamaCppBackend` ejecuta correctamente `Qwen1.5-MoE-A2.7B` en la GTX 1650 Ti 4GB.
- **GATE D:** `LlamaCppBackend` mantiene $\ge 10.0\text{ tok/s}$ sostenidos bajo condiciones reproducibles.
- **GATE E:** Benchmark común completado con métricas físicas de los tres motores.
- **GATE F:** Coordinator, Skills, RAG y Memory operan indistintamente con cualquiera de los 3 backends.
- **GATE G (Para MoE Engine):** Alcanzar $\ge 18.64\text{ tok/s}$ para declarar ventaja frente a llama.cpp.
- **GATE H:** Cero degradación de estabilidad, exactitud numérica o fugas de memoria.

### 9.2. Criterios de Abandono (Kill Criteria)
- Si un backend causa cuelgues del driver NVIDIA (`TDR`), OOM irrecuperable o inestabilidad del sistema, se desactiva inmediatamente.
- Si nuestro MoE Engine en C++/CUDA no logra superar $\ge 18.64\text{ tok/s}$ tras la fase de compilación nativa, se preserva como herramienta de análisis/telemetría y `llama.cpp` se ratifica como backend MoE principal definitivo.

---

## 10. Clasificación de Tareas y Protocolo de Colaboración Gemini

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MATRIZ DE COLABORACIÓN Y COMPLEJIDAD                            │
├────────────┬──────────────────┬────────────────────────┬───────────────────────────────┤
│ Nivel      │ Modelo Asignado  │ Tipo de Tarea          │ Módulos Involucrados          │
├────────────┼──────────────────┼────────────────────────┼───────────────────────────────┤
│ **LOW**    │ Gemini 3.7 Low   │ Documentación, tests   │ `docs/`, `benchmarks/json`,   │
│            │                  │ simples, scripts util  │ reportes markdown             │
├────────────┼──────────────────┼────────────────────────┼───────────────────────────────┤
│ **MEDIUM** │ Gemini 3.7 Medium│ Interfaces abstractas, │ `providers/base.py`,          │
│            │                  │ adaptadores HTTP/SSE,  │ `providers/llamacpp_provider`,│
│            │                  │ wrappers de procesos   │ `core/engine.py`              │
├────────────┼──────────────────┼────────────────────────┼───────────────────────────────┤
│ **HIGH**   │ Gemini 3.7 High  │ CUDA kernels, memoria  │ `core/moe/`, CUDA streams,    │
│            │                  │ física, DMA Pinned,    │ VRAM allocation, sincronía    │
│            │                  │ optimizaciones C++ MoE │ GPU/CPU, hot-swap libre de OOM│
└────────────┴──────────────────┴────────────────────────┴───────────────────────────────┘
```

---

## 11. Orden Exacto de Implementación (Secuencia de Bajo Riesgo)

La estrategia de ejecución sigue el principio de **cero disrupción**: primero se valida la integración de `llama.cpp` de forma aislada sobre las interfaces existentes sin refactorizaciones prematuras, y solo tras verificar la funcionalidad completa de AS-Core se consolida la abstracción.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SECUENCIA DE IMPLEMENTACIÓN OPTIMIZADA                          │
├────────┬────────────────────────────────────────────────────────────────┬──────────────┤
│ Paso   │ Tarea / Hito Técnico                                           │ Nivel Gemini │
├────────┼────────────────────────────────────────────────────────────────┼──────────────┤
│ **P0** │ Auditoría final de interfaces existentes (`providers/base.py`) │ LOW          │
│ **P1** │ `LlamaCppProvider` mínimo y aislado (Daemon Windows CUDA)      │ MEDIUM       │
│ **P2** │ Probar `Qwen1.5-MoE-A2.7B` real desde AS-Core (`EngineManager`)| MEDIUM       │
│ **P3** │ Validar Coordinator + Skills + RAG + Memory sobre llama.cpp    │ MEDIUM       │
│ **P4** │ Benchmark comparativo real (Gemma E2B / OLMoE / llama.cpp)     │ MEDIUM       │
│ **P5** │ Consolidar / refactorizar formalmente `InferenceBackend`        │ MEDIUM       │
│ **P6** │ Perfilado detallado y optimización de nuestro MoE Engine       │ HIGH         │
│ **P7** │ Benchmark final MoE Engine vs llama.cpp (Evaluación Gate G)    │ HIGH         │
└────────┴────────────────────────────────────────────────────────────────┴──────────────┘
```
