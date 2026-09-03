# 🗺️ MASTER ROADMAP — AS CORE
# EVOLUCIÓN DEL COGNITIVE RUNTIME

> DOCUMENTATION STATUS: UPDATED AGAINST CURRENT CODEBASE
> DATE: 2026-08-29

AS-Core evoluciona hacia un **Runtime Cognitivo Local** centrado en proyectos, combinando memoria, documentos, contexto, orquestación determinista y ejecución desacoplada.

---

## 🏛️ ETAPAS HISTÓRICAS DE FUNDACIÓN (COMPLETADAS)

### ✅ Fase 1 — Core Runtime, RAG & Skills (Completada)
*   **[VERIFIED] LiteRT-LM Windows Runtime:** Inferencia local acelerada para modelos densos compactos (Gemma 3n E2B, Gemma 4 E4B).
*   **[VERIFIED] Smart Routing:** Orquestación por roles (Chat, Code, Reasoning, MoE).
*   **[VERIFIED] SSE Streaming & OpenAI API:** Endpoints `/v1/chat/completions`, `/v1/models`, `/v1/status`, `/v1/cancel`.
*   **[VERIFIED] NotebookLM RAG Pipeline (v2):** Ingesta híbrida SQLite/FAISS, chunking AST para Python, markdown jerárquico y adaptativo.
*   **[VERIFIED] Dynamic Capability Registry:** Descubrimiento de primitivas del entorno (Documents, RAG, Git, Terminal).
*   **[VERIFIED] Skill Runtime v1:** Carga dinámica de manifests JSON y prompts en markdown.

### ✅ Fase 2 — Working Memory Layer (Completada)
*   **[VERIFIED] Tablas de Memoria en SQLite:** Variables, tareas con prioridad y observaciones empíricas aisladas por `session_id`.
*   **[VERIFIED] API CRUD de Memoria:** Endpoints bajo `/v1/memory/*`.
*   **[VERIFIED] Inyección en Prompt de Sistema:** Formateo estructurado de la memoria activa en el prompt.
*   **[VERIFIED] Memory UI Drawer:** Panel lateral interactivo en frontend para visualización y depuración en vivo.

### ✅ Fase 3 — Smart Main Agent & Runtime Coordinator (Completada)
*   **[VERIFIED] Runtime Coordinator Manager:** Control de límites de memoria (15 variables, 10 tareas, 20 observaciones).
*   **[VERIFIED] Workflow State Machine:** Transiciones deterministas (`wf_objective`, `wf_phase`, `wf_focus`).
*   **[VERIFIED] PureCoordinator & RuntimeContract:** Pipeline puro desacoplado: compilación de contexto sin efectos secundarios (`PureCoordinator.assemble`) y mutación atómica post-inferencia (`RuntimeStateMutator`).
*   **[VERIFIED] Conversational Continuity:** `DeterministicContinuityResolver` y `DeterministicLanguageDetector` para resolución determinista de elipsis y anclaje de idioma.
*   **[VERIFIED] Intent Gate & Prompt Family Registry:** Filtrado léxico con límites de palabra (`\b`) y centralización de plantillas en `PROMPT_FAMILIES`.

### ✅ Fase 4 — Capa de Proyectos & Persistencia (Completada)
*   **[VERIFIED] Modelado de Proyectos:** Tablas `projects`, `project_chats`, `project_documents` y `project_chat_messages` en SQLite.
*   **[VERIFIED] Scoping de Contexto:** Aislamiento de chats, documentos y mensajes por proyecto.
*   **[VERIFIED] Persistencia de Historial y Autotítulo:** Guardado automático backend-driven de mensajes y generación de título automático.

### ✅ Fase 5 — Multi-Backend Inferencia & Skill Factory Experimental (Completada)
*   **[VERIFIED] LlamaCppProvider:** Daemon aislado `llama-server.exe` sobre CUDA con SSE, polling de salud y auto-puerto.
*   **[VERIFIED] Selector de Modelos UI:** Selector manual vs automático (`AUTO`, `Gemma E2B`, `Gemma E4B`, `OLMoE`, `Qwen MoE`).
*   **[VERIFIED] VRAM Hot-Swapping:** Descarga cruzada del 100% de VRAM en `EngineManager` al alternar entre LiteRT y llama.cpp.
*   **[VERIFIED] Skill Factory Experimental:** Sandbox aislado en `temp_skills/` con ciclo de testing y propuesta de skills sin tocar las oficiales.

### ✅ Fase 6 — Subsistema Knowledge Graph & Relational Reasoning (Completada & Cerrada)
*   **[VERIFIED] Gate 1 — Stabilization & Baseline:** Congelamiento de código de producción previo y suite base de regresión (137 tests GREEN).
*   **[VERIFIED] Gate 2 — Contratos & Tipos Canónicos:** Modelos Pydantic desacoplados (`GraphEntity`, `GraphRelationship`, `GraphQuery`, `GraphQueryResult`, `GraphProvider`) en `runtime/graph/contracts.py`.
*   **[VERIFIED] Gate 3 — Storage & Persistencia SQLite:** Tablas `graph_nodes`, `graph_edges` y `graph_build_status` con aislamiento por `project_id` e idempotencia vía `GraphStore`.
*   **[VERIFIED] Gate 4 — Extracción & Resolución Cross-Document:** `StructuralExtractor`, `normalize_label`, `normalize_key` y `EntityResolver` para resolución unificada de entidades multiorigen.
*   **[VERIFIED] Gate 5 — Query Engine & Bounded Traversal:** `GraphQueryEngine` con BFS acotado (`max_depth`, `max_nodes`, `timeout_seconds`) y corte estricto de ciclos.
*   **[VERIFIED] Gate 6 — Trigger & Formateador Relacional:** `GraphTrigger` léxico y `RelationalContextFormatter` para inyección de contexto Markdown estructurado en el prompt del LLM.
*   **[VERIFIED] Gate 7 — Runtime Coordinator Integration:** Integración opcional, lazy, fail-safe y bounded en el `RuntimeCoordinator` y `/v1/chat/completions`.
*   **[VERIFIED] Gate 8 (8.1 - 8.6) — Ingestión, Hardening, Fidelidad y Cierre:**
    - Hook de ingestión en upload RAG (`GraphIngestionPipeline`) con rollback atómico y fail-safe isolation.
    - Validación canónica con dataset empresarial real (01 a 05): 11/11 relaciones exactas, 0 relaciones falsas (`Carlos -> depende de -> María` = 0).
    - Cierre arquitectónico, multi-chat unificado por proyecto, 10x determinismo y 207 tests de regresión pasando (0 fallos, 0 regresiones).

---

# 🚀 COGNITIVE RUNTIME EVOLUTION (ROADMAP MAESTRO)

El roadmap maestro organiza la evolución cognitiva de AS-Core en ocho bloques estratégicos, separando estrictamente el desarrollo de producto de las líneas de investigación de I+D.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        PRODUCT DEVELOPMENT                             │
│                                                                        │
│  [ETAPA 1] ───► [ETAPA 2] ───► [ETAPA 3] ───► [ETAPA 4] ───► [ETAPA 5] │
│  Hardening      Cognitive      Intelligent     Dynamic         Cognitive│
│  Agent/UI       Interpreter    Model Select    Skills + Test   UI & Obs │
│                                                                        │
│  [ETAPA 6] ───────────────────► [ETAPA 7]                              │
│  Visual Prompt Engineer         Multi-Provider Consolidation           │
└────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────────────────┐
│                        I+D / INVESTIGACIÓN                             │
│                                                                        │
│  [ETAPA 8] MoE Residency Engine vs llama.cpp (Aislado en core/moe/)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 BLOQUES DE DESARROLLO DE PRODUCTO (PRODUCT DEVELOPMENT)

---

### 🛡️ ETAPA 1 — Agent Loop + UI Hardening
- **Objetivo:** Eliminar llamadas espurias a herramientas, sanitizar outputs, blindar el Capability Gate y garantizar la finalización limpia del stream SSE y el ciclo de vida de la UI (sin cursores huérfanos ni estados congelados).
- **Estado:** 🔄 **EN PLANIFICACIÓN / PRIORIDAD INMEDIATA**
- **Dependencias:** Ninguna (base existente).
- **Prioridad:** P0 (Crítica).
- **Riesgo:** 🟡 YELLOW (modificación acotada de runtime y UI existente).
- **Resultado Esperado:** Agent loop robusto con Capability Gate 100% determinista, UI que finaliza limpiamente en el 100% de los casos y suite de tests 100% GREEN.

---

### 🧠 ETAPA 2 — Capa de Interpretación Cognitiva (Cognitive Interpreter)
- **Objetivo:** Construir una capa de interpretación intermedia entre el input del usuario y el coordinador que extraiga intenciones estructuradas, reformule consultas complejas, identifique restricciones explícitas y descomponga objetivos multietapa.
- **Estado:** ⬜ Pendiente (Depende de Etapa 1).
- **Dependencias:** Etapa 1 completada, `RuntimeContract`, `PureCoordinator`.
- **Prioridad:** P1.
- **Riesgo:** 🟡 YELLOW.
- **Resultado Esperado:** Detección de intenciones precisa y contextual sin agregar modelos neuronales pesados para tareas heurísticas.

---

### 🎯 ETAPA 3 — Intelligent Model Selection 2.0
- **Objetivo:** Evolucionar el SmartRouter hacia un selector cognitivo consciente del estado de hardware (VRAM/RAM libre), complejidad de la tarea requerida y capacidades activas, preservando el control manual del usuario (`MANUAL > AUTO`).
- **Estado:** ⬜ Pendiente.
- **Dependencias:** Etapa 1 y Etapa 2.
- **Prioridad:** P1.
- **Riesgo:** 🟡 YELLOW.
- **Resultado Esperado:** Enrutamiento dinámico óptimo entre modelos ligeros (Gemma 2B) y de razonamiento/MoE (Gemma 4B, Qwen MoE) con latencia mínima.

---

### 🏭 ETAPA 4 — Dynamic Skills / Skill Factory + Testing Loop
- **Objetivo:** Consolidar el ciclo de vida completo de las habilidades dinámicas (`draft` $\to$ `testing` $\to$ `candidate` $\to$ `proposal` $\to$ `promotion`), integrando el bucle de edición, testing iterativo y análisis de regresión en el Skills Lab.
- **Estado:** 🔄 Parcialmente implementado en backend (`factory.py`, `temporary.py`) y UI básica.
- **Dependencias:** Etapa 1 y Etapa 2.
- **Prioridad:** P2.
- **Riesgo:** 🟢 GREEN (completamente aislado en sandbox).
- **Resultado Esperado:** Creación, prueba y promoción segura de nuevas habilidades sin riesgo de corrupción de las skills oficiales.

---

### 📊 ETAPA 5 — UI Cognitiva / Observabilidad
- **Objetivo:** Proveer una experiencia de usuario interactiva y transparente que visualice el proceso de pensamiento del runtime (árbol de decisiones del coordinador, trazas de intención, scoring de recuperación RAG y métricas de ejecución por paso).
- **Estado:** ⬜ Pendiente.
- **Dependencias:** Etapa 1 a 4.
- **Prioridad:** P2.
- **Riesgo:** 🟡 YELLOW.
- **Resultado Esperado:** Observabilidad completa del runtime cognitivo en la interfaz gráfica sin penalizaciones de rendimiento.

---

### 🎨 ETAPA 6 — Visual Prompt Engineer
- **Objetivo:** Editor visual interactivo integrado para diseñar, simular, ajustar variables de contexto (`[LANG]`, `{working_memory}`, `{rag_context}`) y realizar linting de prompts de sistema en tiempo real con cálculo de presupuesto de tokens.
- **Estado:** ⬜ Pendiente.
- **Dependencias:** Etapa 4 y 5.
- **Prioridad:** P3.
- **Riesgo:** 🟢 GREEN (herramienta aditiva).
- **Resultado Esperado:** Entorno visual para la ingeniería y calibración de directivas del agente.

---

### 🔌 ETAPA 7 — Multi-Provider Consolidation
- **Objetivo:** Estandarizar la abstracción `InferenceProvider` para soportar de forma homogénea múltiples backends locales y remotos con contratos unificados de cancelación, streaming, health-checks y auto-recuperación de procesos daemon.
- **Estado:** 🔄 Base funcional existente (`litert_cli` + `llamacpp`).
- **Dependencias:** Etapa 1 a 6.
- **Prioridad:** P3.
- **Riesgo:** 🟡 YELLOW.
- **Resultado Esperado:** Capa de proveedores de inferencia intercambiable y tolerante a fallos.

---

## 🔬 LÍNEA DE INVESTIGACIÓN Y EXPERIMENTACIÓN (I+D)

---

### 🧪 ETAPA 8 — MoE Residency Engine R&D vs llama.cpp
- **Objetivo:** Investigar y optimizar el motor propio de residencia dinámica MoE (`core/moe/`) evaluando estrategias de swapping LRU (VRAM $\leftrightarrow$ RAM $\leftrightarrow$ NVMe), cuantización mixta y pre-enrutamiento de capas para modelos MoE grandes en GPUs de consumo (4GB-8GB VRAM).
- **Condición de Promoción a Producción:** Permanecerá estrictamente en el entorno experimental de `core/moe/` hasta demostrar de manera reproducible una superioridad cuantitativa en throughput (tok/s), latencia y estabilidad frente a `llama.cpp` en escenarios reales de Windows.
- **Estado:** 🔬 **I+D ACTIVA / AISLADA DE PRODUCCIÓN**
- **Dependencias:** Pesos GGUF locales, CUDA / PyTorch.
- **Prioridad:** I+D (Línea paralela).
- **Riesgo:** 🔴 RED si se integra a producción; 🟢 GREEN en su sandbox de investigación.
- **Resultado Esperado:** Benchmarks cuantitativos y reportes comparativos frente a `llama.cpp`.
