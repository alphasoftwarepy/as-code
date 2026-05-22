# ROADMAP

AS Code is evolving from a local chat server into a **Local AI Operating Layer** (Offline-first, modular, and extensible alternative to Claude Code, Cursor, and NotebookLM).

## ✅ Phase 1 — Runtime Core & RAG NotebookLM (Completed)

*   **LiteRT-LM Windows Runtime:** GPU-accelerated local inference utilizing Gemma 3.
*   **Smart Routing:** Multi-role orchestration (Chat, Code, Reasoning).
*   **SSE Streaming & OpenAI API:** Drop-in compatibility for client tools (Cline, Continue, etc.).
*   **Hardware-Adaptive Profiles:** Auto-tuning of parameters based on VRAM/CPU capability.
*   **NotebookLM RAG Pipeline (RAG v2):**
    *   Direct RAG ingest via `/api/rag/documents/upload`.
    *   Local embeddings (`BAAI/bge-small-en-v1.5`) + FAISS index + SQLite metadata.
    *   AST parsing for Python (.py), heading hierachy for markdown, and paragraph segmenting for PDFs.
    *   Hybrid retrieval: `alpha * semantic + (1 - alpha) * keyword (BM25)`.
    *   Structured context composition (`NotebookContextBuilder`) with `normal`, `thinking`, and `code` modes.
    *   Prompt stuffing and legacy document sessions completely eliminated.

## 🚧 Phase 2 — Runtime Capability System (Current Focus)

*   **Dynamic Capability Registry:** A central registry detecting local environment capabilities.
*   **Safety & Overrides:** User controls (`capability_overrides` dict) to explicitly block features (e.g. terminal execution).
*   **Hardware & Provider Awareness:** Distinguishing between available, enabled, and health/offline status of each capability.
*   **Dynamic Capabilities API:** `/v1/capabilities` exposing status, scopes, display names, and categories to the client.

## 🔮 Phase 3 — Workspace & Persistence

*   **Premium Workspace UI:** Sidebars, private chat history browser, and visual document managers.
*   **Auto-Save & Session Restore:** Permanent persistence of multi-document conversations.
*   **Shared Memory:** Cross-document search and workspace-level context compilation.
*   **Secure Multi-Tenant Isolation:** User registration, password hashing (bcrypt), JWT tokens, and private folders/databases per user.

## 🔮 Phase 4 — Skill Runtime v1

*   **Skill Manifests:** Declaring metadata, capabilities list, and required versioning.
*   **Prompt Injection Framework:** Dynamic injection of system prompt extensions by active skills.
*   **Scope Checks:** Skill permission verification against available and enabled capability scopes.
*   **Activation Lifecycles:** Active, idle, and deactivated states of local skills.

## 🔮 Phase 5 — Skill Builder

*   **No-Code/Low-Code Creation:** Builder UI to assemble prompts, parameters, and consumed capabilities into custom skills.
*   **Validation Registry:** Verification of capabilities before a skill can be saved or deployed.

## 🔮 Phase 6 — Tools & Execution

*   **System Tools:** Core local capability execution (git status/diff, filesystem read/write, terminal execution).
*   **Model Context Protocol (MCP):** Integration of external client/server tools.
*   **Security Sandboxing:** Safe terminal execution boundaries.

## 🔮 Phase 7 — Marketplace / Import / Export

*   **Workspace Export:** Shareable workspace and skill packages (manifests + assets).
*   **Local Skills Hub:** Offline import/export and registry database for team sharing.

## 🔮 Phase 8 — Claude/MCP Translation Layer

*   **Adapter Layer:** API translation layer ensuring that skills/tools designed for Claude Code or MCP operate seamlessly within the AS Code local execution runtime.
