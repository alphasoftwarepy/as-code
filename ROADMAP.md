# ROADMAP

AS Code is evolving from a local chat server into a **Unified Smart Main Agent Runtime** (Offline-first, modular, and extensible alternative to Claude Code, Cursor, and NotebookLM).

---

## ✅ Phase 1 — Core Runtime, RAG & Skills (Completed)

*   **LiteRT-LM Windows Runtime:** GPU-accelerated local inference utilizing Gemma 3.
*   **Smart Routing:** Multi-role orchestration (Chat, Code, Reasoning).
*   **SSE Streaming & OpenAI API:** Drop-in compatibility for client tools (Cline, Continue, etc.).
*   **Hardware-Adaptive Profiles:** Auto-tuning of parameters based on VRAM/CPU capability.
*   **NotebookLM RAG Pipeline (RAG v2):**
    *   Direct RAG ingest via `/api/rag/documents/upload`.
    *   Local embeddings (`BAAI/bge-small-en-v1.5`) + FAISS index + SQLite metadata.
    *   AST parsing for Python (.py), heading hierarchy for markdown, and structure-agnostic adaptive semantic segmenting (paragraph → sentence → char fallback) for PDFs, Word documents, and text.
    *   Hybrid retrieval: `alpha * semantic + (1 - alpha) * keyword (BM25)`.
    *   Structured context composition (`NotebookContextBuilder`) with `normal`, `thinking`, and `code` modes.
*   **Runtime Capability Registry:** Dynamic discovery of environment primitives (Git, Terminal, Documents, RAG).
*   **Skill Runtime v1:** Discoverable JSON manifests and dynamic system prompt injection framework.

---

## ✅ Phase 2 — Working Memory Layer (Completed)

A structured short-term cognitive scratchpad to keep track of agent goals, variables, and observations, fully isolated by session.

*   **Session Isolation:** Explicit `session_id` on all memory tables for future-proof multi-chat / VSCode tab isolation.
*   **Runtime-Native Protocol:** Simple endpoints (`/v1/memory/*`) for CRUD operations on variables, tasks, and observations.
*   **Task Management:** Priority-aware task list (P0, P1...) allowing the agent to sort objectives.
*   **Fact Tracking (Observations):** Observation logs categorized by source (`user`, `system`, `rag`, `capability`) for explanation and debugging.
*   **System Prompt Injection:** Injects formatted memory state directly into the system context in the correct cognitive order: `base_prompt` → `skill_prompt` → `Working Memory` → `RAG Context` (user message) → `History` → `User Message`.
*   **Event-Driven UI:** Collapsible Memory Drawer showing real-time state, updating only on interactions to save resources.

---

## ✅ Phase 3 — Smart Main Agent Foundation & Runtime Coordinator (Completed)

Developing the coordinator, deterministic state machines, task auto-progression, and recommended skills engine.

*   **Runtime Coordinator Manager:** Central orchestrator managing cognitive limits (15 vars, 10 tasks, 20 observations) to prevent token pollution.
*   **Workflow State Machine:** Deterministic transition tracker (`wf_objective`, `wf_phase`, `wf_focus`) with automatic task progression based on user intent.
*   **Skill Recommendation Engine:** Suggestions for switching/activating compatible runtime skills based on intent and phase.
*   **Unified UI Integration:** Beautiful Workflow Header badge, active Phase pill, Current Focus info, and clickable Suggested Skill chips.
*   **Output Stabilization Layer:** Line-buffering stdout sanitizer in provider stream to discard initialization logs and raw parameter echoes.
*   **Backend Parameter Presets:** Semantic config presets (PRECISE, BALANCED, CREATIVE) mapped automatically to active pipelines/skills to prevent parameter drift.
*   **Working Memory UX Clarifications:** Collapsible settings details layout, preset selector dropdown, operational info card, and section headers hover tooltips.
*   **Runtime Hardening (Fase 1):**
    *   **RuntimeContract**: Immutable model representing request state and session history.
    *   **ContextManifest**: Serializable intermediate snapshot of prompts, budgets, and RAG details.
    *   **PureCoordinator**: Stateless prompt assembly engine ensuring zero database writes during the prompt compilation phase.
    *   **RuntimeStateMutator**: Post-inference atomic database mutator separating context rendering from side effects.
    *   **Conversational Continuity (Continuity Fix)**: Pure deterministic `DeterministicContinuityResolver` and `DeterministicLanguageDetector` with `LightweightStateStore`. Resolves follow-up queries using entropy-based non-positional carryover, sticky language detection (preventing language oscillation), and explicit/implicit self-sufficiency triggers.
    *   **Excel Parsing Support**: Seamless parsing of `.xlsx` and `.xlsm` formats via `openpyxl` into clean, structured Markdown tables, with row and column caps to protect the context window.
    *   **Skill Suggestions Engine Hardening**: Reordered fallback skills to prioritize `"programming"`, expanded keywords to capture technical intents, and removed the early break limit to allow up to 3 recommendations.
    *   **Working Memory Prompt Sanitation**: Filtered all internal `"wf_*"` variables from the system prompt block to prevent token bloat, and cleaned up the runtime context block by omitting phase/focus info and generic objectives.
    *   **Session Isolation & Reset UX**: Prevents workflow and variable contamination across sessions by using dynamic session IDs, resetting client UI state on clear/upload, and purging old objective/phase data when the active skill transitions.

---

## ✅ Phase 3.5 — Agent Control Loop & Native Call Parser (Completed)

Developing the decision-making loop and output syntax parsing to allow the unified model to orchestrate its own actions.

*   **Native Protocol Parser:** Stream-aware XML or JSON tag listener detecting capability execution requests (e.g. `{"capability": "git", "action": "status", "params": {}}`).
*   **Server-Side Agent Loop:** Intercepting capability calls, suspending generation, executing the action, and feeding outputs back into the chat loop.
*   **Cognitive Prompt Tuning:** Formatting base instructions to guide the model on when to write to memory and when to call tools.

---

## 
---

## Phase 3.6 — Intent Routing & Prompt Resolution (Completed)

A cycle of surgical fixes to the intent routing, persona resolution, and workflow instrumentation layers, validated through production logs and automated regression tests.

*   **Session-Scoped RAG (Active Document Context):** RAG retrieval strictly filtered by active `session_id`. Document uploads preserve the current session. RAG is disabled when no documents exist in the session.
*   **User Intent Gate:** `analyze_intent` returns an empty list immediately when the user message contains no skill-specific keywords. Word-boundary matching (`\b`) blocks sub-word false positives.
*   **Prompt Family Registry:** New `runtime/coordinator/prompts.py` centralizes all prompt templates under `PROMPT_FAMILIES`. Resolution driven by `prompt_family` in each skill `manifest.json`. No hardcoded skill lists in the coordinator.
*   **BUSINESS_PROMPT decoupled from analytical structure:** Removed mandatory DIAGNOSTICO / ANALISIS / ACCION sections. Now defines domain identity only, letting the LLM adapt output format to the actual task type.
*   **WorkflowContinuationResolver (v1_passthrough):** New architectural extension point separating workflow continuity from retrieval continuity. Emits `[WORKFLOW-TRACE]` logs for evidence collection before continuity rules are derived.

> See [`dev-notes/STABLE.md`](dev-notes/STABLE.md) for validated invariants that must not be modified without impact analysis.

🚧 Phase 4 — Capability Execution (using `capability.execute()`) (Next)

Activating capabilities by providing execution primitives directly within capability classes.

*   **Base Interface Extension:** Adding an async `execute(action, params)` method to `BaseCapability`.
*   **Local Terminal Command Runner:** Running shell processes safely, handling outputs, timeouts, and return codes.
*   **Local Git Interface:** wrapper to fetch diffs, checkout branches, and stage commits.
*   **Scope Security Boundaries:** Enforcing permission boundaries before letting a skill invoke a capability.

---

## 🔮 Phase 5 — Human-in-the-Loop (HITL) Queue

Adding user confirmation gates for high-impact or destructive operations.

*   **Suspended Execution Queue:** FastAPI state queue keeping pending commands.
*   **HITL API Endpoints:** Endpoints `/v1/capabilities/pending` and `/v1/capabilities/confirm`.
*   **Interactive UI Modal:** Consent dialog inside the browser UI to allow the user to modify or approve commands.

---

## 🔮 Phase 6 — Workspace, Projects, Chats & Knowledge Isolation

Isolating files, conversations, and context variables by active directories, supporting shared project-level documents and private chat attachments.

*   **Workspace & Session Models:** Database tables to map active projects and group sessions.
*   **Retrieval Scoping:** Filtering FAISS and BM25 queries to prevent cross-project knowledge contamination.

---

## 🔮 Phase 7 — IDE / VSCode Integration

Exposing the Unified Agent Runtime to external developer editors.

*   **Workspace Sync API:** Syncing working folders, cursor positions, and open file buffers.
*   **Cline / Continue Adapters:** Formatting local routes to act as custom providers for standard extensions.

---

## 🔮 FASE EXTRA — Multi-User Sharing, Team VPN & Authentication

Security layer for team servers and shared GPU homelabs over VPN, preventing data leakage.

*   **Secure Authentication:** Secure logins, password hashing (bcrypt), and JWT tokens.
*   **Private Data Isolation:** Database filters to partition chats and knowledge bases by `user_id`.
