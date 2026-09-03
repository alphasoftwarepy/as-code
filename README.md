# AS-Core: Runtime Cognitivo

**By Alpha Software**

**AS-Core** is a local-first **Cognitive Runtime** (Runtime Cognitivo) and workspace centered around projects that combines multi-backend neural inference, dynamic MoE management, memory, documents, context, and execution to help people and organizations complete real work in a private, auditable, and fully owned environment.

> **AS-Core is a cognitive runtime capable of coding, deep reasoning, and autonomous execution.** It manages project context, persists working memory, routes across specialized models/experts, and executes work locally.

---

## 🧠 ¿Qué es un Runtime Cognitivo? (What is a Cognitive Runtime?)

A conventional AI chat interface or API wrapper merely passes text to a single model. A **Cognitive Runtime** (*Runtime Cognitivo*) is a complete execution environment that orchestrates the entire lifecycle of local intelligence:

* **Heterogeneous Model Execution:** Dynamically switches and routes across inference backends (**LiteRT-LM** for ultra-efficient Windows execution and **LLaMA.cpp** for GGUF/MoE models) with zero memory leaks.
* **MoE Dynamic Residency & Expert Routing:** Executes large Mixture of Experts models that exceed physical VRAM by orchestrating layer-by-layer hotset caching and RAM/VRAM offloading.
* **Structured Working Memory:** Maintains session-aware and project-aware short/long-term memory tables (variables, prioritized tasks, observations) injected directly at the cognitive system level.
* **Hierarchical Context & Retrieval (RAG v2):** Ingests, AST-chunks, and retrieves information with hybrid vector (FAISS) + keyword (BM25) search.
* **Deterministic Execution & Observability:** Implements an auditable agent execution loop with strict tracing (`RAG-SCOPE`, `ROUTING-TRACE`, `PROMPT-TRACE`).

---

## 🎯 Mission Statement

AS-Core provides a local-first cognitive workspace centered around projects that combines memory, documents, context, and execution to help people and organizations complete real work in a private, auditable, and fully owned environment.

---

## 🛠️ Core Principles

* **Project Ownership:** Every piece of knowledge belongs somewhere. Documents, chats, memory, tasks, and executions belong strictly to a project. There is no global retrieval contamination or context leakage.
* **Context Ownership:** What you see is what the system can access. No hidden context, no global contamination.
* **Local First:** The system remains usable offline on consumer hardware (e.g., 16 GB RAM). Context precision is prioritized over massive context windows.
* **Determinism & Observability:** Every runtime decision is explainable. The system implements strict tracing: `RAG-SCOPE`, `SKILL-TRACE`, `WORKFLOW-TRACE`, `ROUTING-TRACE`, and `PROMPT-TRACE`.
* **Execution Over Conversation:** Conversation is not the final product—execution is. AS-Core helps users organize information, analyze documents, process files, manage tasks, and execute workflows.

---

## 🚀 Current Status & What's New

AS-Core has evolved from a local chat server into an extensible, project-centric cognitive workspace runtime.

* **Multi-Backend & MoE Engine (Latest):** 
  - **LLaMA.cpp Backend:** Full integration supporting `.gguf` models, K-quants, and custom GPU offloading (`n_gpu_layers`).
  - **MoE Architecture (Mixture of Experts):** Execution of MoE models (e.g., OLMoE 1B-7B, Qwen1.5-MoE) on consumer GPUs through dynamic expert residency.
  - **VRAM & RAM Pools:** High-frequency expert hotsets cached in VRAM (Pool B2) with background staging in RAM (Pool B3) and predictive LRU swapping (B4).
  - **Routing Tracer & Frequency Tracker:** Real-time observability of token-to-expert pathways and activation patterns.
* **Phase 1 (Core & RAG NotebookLM):** LiteRT-LM Windows inference (GPU accelerated), OpenAI-compatible API, dynamic capability registry, skill prompt injection, and hybrid semantic/keyword retrieval vector pipeline.
* **Phase 2 (Working Memory Layer):** Runtime-native CRUD memory tables (variables, tasks with priority, observations), session-based isolation (`session_id`), cognitive prompt injection in system prompt, and event-driven UI panel.
* **Phase 3 (Smart Main Agent Foundation & Runtime Hardening):** Unified Runtime Coordinator managing memory limits, deterministic workflow transitions, and skill suggestions. Includes output stabilization, backend presets, and runtime hardening (immutable `RuntimeContract`/`ContextManifest` flow).
* **Phase 3.5 & 3.6 (Agent Loop & Intent Gate):** Server-side agent loops, native execution protocol parsing (`capability.execute()`), session-scoped RAG (Active Retrieval Scope), intent gate keyword boundaries (`\b`), and prompt family registry.
* **Phase 4 (Project Layer):** Scoping chats, documents, and memory under unified `project_id` boundaries.
* **Phase 6 / Knowledge Graph Subsystem (Latest):** 
  - **Relational Knowledge Graph:** Optional, fail-safe, project-scoped relational reasoning subsystem.
  - **Deterministic Extraction & Resolution:** Extracts structural S-V-O relationships and unifies entities cross-document without false relations.
  - **Bounded Traversal (BFS):** Strict control over cognitive exploration (`max_depth`, `max_nodes`, `timeout_seconds`) with cycle interruption.
  - **RAG + Graph Dual Retrieval:** Documents uploaded via RAG trigger incremental, atomic Graph persistence with complete provenance (`source_doc_id`).
  - **Prompt Injection:** Seamless Markdown formatting of multi-hop relational context injected directly into the LLM system prompt.

---

## ⚡ Key Features

* **Multi-Backend Engine:** Native support for both **LiteRT-LM** (`.litertlm`) and **LLaMA.cpp** (`.gguf`).
* **MoE Dynamic Residency:** Run massive Mixture of Experts models exceeding your GPU VRAM without crashing.
* **Hardware-Adaptive Profiles:** Auto-tunes settings (VRAM pools, GPU offload layers, thread allocation) to match system specs.
* **Working Memory Layer:** Persistent, session-aware short-term memory (variables, tasks, observations) injected dynamically at the SYSTEM level.
* **Knowledge Graph Layer:** Local SQLite relational graph (`graph_nodes`, `graph_edges`) providing deterministic cross-document entity connections and structural reasoning.
* **RAG NotebookLM (RAG v2):** Multi-stage local pipeline: parses, AST-aware chunking, generates local embeddings, stores metadata in SQLite + vectors in FAISS, and executes hybrid retrieval.
* **Low-Overhead Hot-Swapping:** Intelligent model loading and idle timeout unloads with zero memory leaks.
* **Browser-First UI:** Premium browser interface with real-time streaming and direct document drop zone.
* **OpenAI-Compatible API:** Serve as a backend for VS Code extensions (Cline, Continue, Roo Code) and external tooling.

---

## 📸 Screenshots

### Local AI Chat & Workspace
![AS Core UI](screenshots/ui-chat.png)

### Features shown
- Multi-model routing & MoE execution
- GPU acceleration & VRAM telemetry
- Local inference & real-time streaming
- Browser-based UI & memory panel
- Document upload and vector indexing

---

## 💻 Hardware Philosophy

AS-Core is built for "Real Hardware"—the laptops and desktops people actually own. While a dedicated GPU is recommended, our architecture is designed to remain responsive on mid-range systems.

- **Optimized for:** Windows 10/11
- **Focus:** Maximum performance per watt/GB on 16 GB RAM laptops with consumer GPUs.

---

## 🏗 Architecture Summary

AS-Core uses a modular architecture built on top of FastAPI, LiteRT-LM, and LLaMA.cpp. It acts as an intelligent routing and execution layer for local models, abstracting away the complexity of VRAM management, expert hotset caching, and hardware-specific configurations, while exposing a standard OpenAI-compatible REST API.

---

## 🛠 Installation

### Prerequisites
- Windows 10/11
- PowerShell 5.1+ or PowerShell Core
- Python 3.10+
- (Optional for LiteRT) `litert-lm` CLI installed via `uv`:
  ```powershell
  uv tool install litert-lm
  ```
- (Optional but recommended) Compatible NVIDIA GPU drivers

### Setup

Clone the repository and run the setup script:

```powershell
git clone https://github.com/alphasoftwarepy/as-core.git
cd as-core
.\scripts\install.ps1
```

The install script will:
1. Create and activate a Python virtual environment (`venv`)
2. Install all dependencies (`pip install -r requirements.txt`)
3. Create required directories (`models/`, `uploads/`, `logs/`, `cache/`)
4. Copy `.env.example` → `.env` with sensible defaults
5. Detect GPU availability and set the appropriate backend

---

## 🧠 Model Setup & Supported Backends

AS-Core uses a **Role-Based Architecture**. The internal logic maps cognitive roles to specialized models and backends:

| Role | Purpose | Supported Backends | Model Formats |
|---|---|---|---|
| **Chat / General** | General conversation and planning | LiteRT-LM / LLaMA.cpp | `.litertlm` / `.gguf` |
| **Code** | Technical tasks and programming | LLaMA.cpp / LiteRT-LM | `.gguf` / `.litertlm` |
| **Reasoning / MoE** | Deep analysis, MoE multi-expert routing | LLaMA.cpp (MoE Engine) | `.gguf` (e.g. OLMoE, Qwen-MoE) |

### Option A: LiteRT-LM Models (`.litertlm`)
1. Create the directory: `models\gemma\`
2. Download `.litertlm` models from [HuggingFace — litert-community](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm).
3. Place files at: `models\gemma\gemma-3n-E2B-it-int4.litertlm`

### Option B: GGUF & MoE Models (`.gguf`)
1. Place your `.gguf` models in the `models/` directory.
2. For MoE models (such as `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` or `qwen1.5-moe-a2.7b-q4_k_m.gguf`), configure their path and backend in `config.yaml` or through the UI.

---

## 🏃‍♂️ Running the Project

Start the local runtime using the provided script:

```powershell
.\scripts\run.ps1
```

Once running, open your browser at `http://localhost:8000`.

---

## 📄 Document Ingest & RAG NotebookLM (v2)

AS-Core includes a full vector-search RAG pipeline running 100% locally.

### Supported Formats & Chunking Strategy
- **Python / JS / TS / Go**: AST-aware chunking by function/class boundaries with symbol metadata.
- **Markdown / RST**: Heading hierarchy chunking (`#`, `##`, ...) with adaptive fallback.
- **PDF / TXT / DOCX**: Structure-agnostic adaptive semantic chunking.
- **XLSX / XLSM**: Spreadsheet parsing formatted into Markdown tables.

### Activation
Add to your `.env`:
```ini
ASCODE_ENABLE_RAG_MODE=true
```

### Ingesting Documents via API

```bash
# Chat pipeline (documents, notes, PDFs)
curl -X POST http://localhost:8000/api/rag/documents/upload \
  -F "file=@README.md" -F "pipeline=chat"

# Code pipeline (source files)
curl -X POST http://localhost:8000/api/rag/documents/upload \
  -F "file=@api/main.py" -F "pipeline=code"
```

---

## 🔌 API Endpoints

Once running, the API is available at `http://localhost:8000`. Interactive Swagger documentation is available at `/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the browser UI |
| `/health` | GET | Health check and system readiness |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/v1/models` | GET | List available local models and active roles |
| `/v1/status` | GET | Telemetry (GPU, VRAM, RAM, active backend) |
| `/v1/cancel` | POST | Cancel in-progress generation |
| `/v1/providers` | GET | List registered inference providers (LiteRT, LLaMA.cpp) |
| `/v1/capabilities` | GET | Dynamic runtime capabilities status |
| `/v1/memory` | GET | Working memory snapshot (variables, tasks, observations) |
| `/v1/memory/variables` | POST/DELETE | CRUD variables in working memory |
| `/v1/memory/tasks` | POST/PATCH/DELETE | CRUD tasks in working memory (status, priority, title) |
| `/v1/memory/observations` | POST/DELETE | CRUD observations with source provenance |
| `/v1/memory/reset` | POST | Clear all working memory for the session |
| `/api/rag/documents/upload` | POST | Upload & ingest document (NotebookLM RAG v2) |
| `/api/rag/documents` | GET | List indexed RAG documents with chunk counts |
| `/api/rag/documents/{id}` | DELETE | Delete document + chunks + physical files + vectors |

---

## 🧠 Runtime Capabilities & IDE Connectors

AS-Core evaluates system capabilities lazily. The UI queries `GET /v1/capabilities` to render available controls dynamically.

Because AS-Core exposes an **OpenAI-compatible API**, you can connect it directly to IDE extensions like **Cline**, **Continue**, or **Roo Code** by setting the base URL to `http://localhost:8000/v1`.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Run test suites with:

```powershell
pytest tests/
```

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details. Commercial use, modification, and redistribution are allowed. Attribution to Alpha Software is required.
