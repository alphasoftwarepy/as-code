# AS Core

**By Alpha Software**

AS Core is a local-first cognitive workspace centered around projects that combines memory, documents, context, and execution to help people and organizations complete real work in a private, auditable, and fully owned environment.

> **AS Core is a cognitive workspace capable of coding among many other tasks.** Programming remains a first-class supported capability and skill domain, but it is no longer the sole definition of the product. The system manages project context, persists working memory, and executes work locally.

---

## 🎯 Mission Statement

AS Core is a local-first cognitive workspace centered around projects that combines memory, documents, context, and execution to help people and organizations complete real work in a private, auditable, and fully owned environment.

---

## 🛠️ Core Principles

*   **Project Ownership:** Every piece of knowledge belongs somewhere. Documents, chats, memory, tasks, and executions belong strictly to a project. There is no global retrieval contamination or context leakage.
*   **Context Ownership:** What you see is what the system can access. No hidden context, no global contamination.
*   **Local First:** The system remains usable offline on consumer hardware (e.g., 16 GB RAM). Context precision is prioritized over massive context windows.
*   **Determinism & Observability:** Every runtime decision is explainable. The system implements strict tracing: `RAG-SCOPE`, `SKILL-TRACE`, `WORKFLOW-TRACE`, and `PROMPT-TRACE`.
*   **Execution Over Conversation:** Conversation is not the final product—execution is. AS Core helps users organize information, analyze documents, process files, manage tasks, and execute terminal workflows.

---

## Current Status

AS Core is in an active development stage, evolving from a local chat server into an extensible, project-centric cognitive workspace runtime.

Core architecture and **Phases 1, 2 & 3** are fully completed:
- **Phase 1 (Core & RAG NotebookLM):** LiteRT-LM Windows inference (GPU accelerated), OpenAI-compatible API, dynamic capability registry, skill prompt injection, and hybrid semantic/keyword retrieval vector pipeline.
- **Phase 2 (Working Memory Layer):** Runtime-native CRUD memory tables (variables, tasks with priority, observations), session-based isolation (`session_id`), cognitive prompt injection in system prompt, and event-driven UI panel.
- **Phase 3 (Smart Main Agent Foundation & Runtime Hardening):** Unified Runtime Coordinator managing memory limits, deterministic workflow transitions, and skill suggestions. Includes output stabilization, backend presets, and runtime hardening (immutable `RuntimeContract`/`ContextManifest` flow).
- **Phase 3.5 & 3.6 (Agent Loop & Intent Gate):** Server-side agent loops, native execution protocol parsing (`capability.execute()`), session-scoped RAG (Active Retrieval Scope), intent gate keyword boundaries (`\b`), and prompt family registry.

Current focus:
- **Phase 4 (Project Layer):** Scoping chats, documents, and memory under unified `project_id` boundaries.

---

## 🚀 Key Features

*   **LiteRT-LM Runtime:** Ultra-optimized inference engine for Windows hardware.
*   **Hardware-Adaptive Profiles:** Auto-tunes settings (such as models and VRAM limits) to match your system's specs.
*   **Browser-First UI:** Premium, minimal browser interface with direct document drop zone.
*   **OpenAI-Compatible API:** Serve as a backend for VS Code extensions (Cline, Continue, etc.) and other clients.
*   **RAG NotebookLM (RAG v2):** Multi-stage local pipeline: parses, chunks (AST-aware), generates local embeddings, stores metadata in SQLite + vectors in FAISS, and executes hybrid retrieval.
*   **Working Memory Layer (Phase 2):** Persistent, session-aware short-term memory (variables, tasks, observations) injected dynamically at the SYSTEM level.
*   **Structured Context Builder:** Composes retrieval context dynamically by grouping chunks under `## CONTEXT FROM DOCUMENTS` by file and section.
*   **Low-Overhead Hot-Swapping:** Intelligent model loading and idle timeout unloads.

---

## 📸 Screenshots

### Local AI Chat

![AS Core UI](screenshots/ui-chat.png)

### Features shown
- Multi-model routing
- GPU acceleration
- Local inference
- Real-time streaming
- Browser-based UI
- Document upload panel

---

## 💻 Hardware Philosophy

AS Core is built for "Real Hardware"—the laptops and desktops people actually own. While a dedicated GPU is recommended, our architecture is designed to remain responsive on mid-range systems.

- **Optimized for:** Windows 10/11
- **Focus:** Maximum performance per watt/GB on 16 GB RAM laptops.

---

## 🏗 Architecture Summary

AS Core uses a modular architecture built on top of FastAPI and LiteRT-LM. It acts as an intelligent routing and execution layer for local models, abstracting away the complexity of VRAM management and hardware-specific configurations, while exposing a standard OpenAI-compatible REST API.

Document context injection is handled as a thin layer between the API and the engine — zero changes to inference infrastructure.

---

## 🛠 Installation

### Prerequisites
- Windows 10/11
- PowerShell
- Python 3.10+
- `litert-lm` CLI installed via `uv`:
  ```powershell
  uv tool install litert-lm
  ```
- (Optional but recommended) Compatible GPU drivers

### Setup

Clone the repository and run the setup script:

```powershell
git clone https://github.com/alphasoftwarepy/as-code.git
cd as-code
.\scripts\install.ps1
```

The install script will:
1. Create and activate a Python virtual environment
2. Install all dependencies (`pip install -r requirements.txt`)
3. Create required directories (`models/gemma/`, `uploads/`, `logs/`, `cache/`)
4. Copy `.env.example` → `.env` with sensible defaults
5. Detect GPU availability and set the appropriate backend

### Python Dependencies

The full dependency list is in `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn` | API server |
| `pydantic`, `pydantic-settings` | Config & models |
| `pypdf` | PDF parsing (document upload) |
| `python-docx` | DOCX parsing (document upload) |
| `openpyxl` | Excel parsing (document upload) |
| `psutil` | System monitoring |
| `pyyaml` | Config file parsing |
| `httpx` | HTTP client |
| `sqlalchemy` | SQLite persistence (RAG metadata) |
| `sentence-transformers` | Local embeddings — BAAI/bge-small-en-v1.5 (RAG v2) |
| `faiss-cpu` | Vector search index (RAG v2) |
| `rank-bm25` | Keyword retrieval for hybrid RAG (RAG v2) |
| `numpy` | Embedding array operations (RAG v2) |

---

## 🧠 Manual Model Setup (Important)

AS Core uses a **Role-Based Architecture**. The internal logic doesn't care about specific model names, only about the role the model plays.

| Role           | Purpose                                      | Model File (LiteRT-LM) |
|----------------|----------------------------------------------|------------------------|
| **Chat**       | General conversation and planning            | `gemma-3n-E2B-it-int4.litertlm` |
| **Code**       | Technical tasks and programming              | `gemma-3n-E2B-it-int4.litertlm` |
| **Reasoning**  | Deep analysis and complex architecture       | `gemma-3n-E2B-it-int4.litertlm` |

> [!IMPORTANT]
> The current engine is ultra-optimized for the **`.litertlm`** (LiteRT-LM) format. You can swap models in `config.yaml`, but ensure they follow this specific encoding for maximum performance on Windows hardware.

**Setup steps:**

1. Create the directory: `models\gemma\`
2. Download the `.litertlm` file from [HuggingFace — litert-community](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm).
3. Place it at: `models\gemma\gemma-3n-E2B-it-int4.litertlm`
4. Run the server — the runtime detects and registers the roles automatically.

---

## 🏃‍♂️ Running the Project

Start the local server using the provided script:

```powershell
.\scripts\run.ps1
```

This will activate the environment, start the FastAPI server, and output logs cleanly.

Once running, open your browser at `http://localhost:8000`.

---

## 📄 Document Ingest (RAG)

AS Core supports uploading documents to chat with their contents. This works entirely locally — no cloud involved.

### Supported formats
- **TXT** — plain text files
- **PDF** — text-based PDFs (not scanned images)
- **DOCX** — Microsoft Word documents
- **XLSX / XLSM** — Excel spreadsheet files (formatted into Markdown tables)

### How to use
1. Open the browser UI at `http://localhost:8000`
2. Drag a file into the **Documentos** panel at the bottom, or click **+ Subir**
3. Once uploaded, all subsequent messages in the session will include the document's content as context
4. Click **🗑 Limpiar** to remove documents from the session

### API (for external clients)

```http
POST /api/documents/session          → Create a session, returns session_id
POST /api/documents/upload?session_id=<id>  → Upload a file (multipart/form-data)
GET  /api/documents/<session_id>     → List documents in session
DELETE /api/documents/<session_id>   → Clear session
```

Include `X-Document-Session-Id: <session_id>` as a header in your `/v1/chat/completions` requests to activate context injection.

---

## 🧠 RAG NotebookLM Pipeline (v2)

AS Core includes a full vector-search RAG pipeline for deeper, more precise document-aware conversations. It runs 100% locally.

### Activation

Add to your `.env`:
```
ASCODE_ENABLE_RAG_MODE=true
```

### What it does differently from RAG v1

| | RAG v1 | RAG v2 |
|---|---|---|
| Storage | In-memory sessions | SQLite (persistent) |
| Retrieval | Full text injection | FAISS vector search + BM25 hybrid |
| Context | Truncated raw text | Hierarchy-aware grouped context |
| Code files | ❌ | ✅ AST chunking by function/class |
| Relevance | None | Cosine similarity score per chunk |

### Supported file types & chunking strategy

| Format | Strategy |
|---|---|
| `.py` | AST — by function/class boundaries with symbol metadata |
| `.md`, `.rst` | By heading hierarchy (`#`, `##`, …) with adaptive fallback |
| `.js`, `.ts`, `.go`, etc. | By function/class (regex) |
| `.pdf`, `.txt`, `.docx` | Structure-agnostic adaptive semantic (paragraph → sentence → char fallback) |

### Upload a document

```bash
# Chat pipeline (documents, notes, PDFs)
curl -X POST http://localhost:8000/api/rag/documents/upload \
  -F "file=@README.md" -F "pipeline=chat"

# Code pipeline (source files)
curl -X POST http://localhost:8000/api/rag/documents/upload \
  -F "file=@api/engine.py" -F "pipeline=code"
```

### Chat with RAG context

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-Enable-RAG: true" \
  -H "X-Mode: normal" \
  -H "X-Pipeline: chat" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain GPU fallback"}],"model":"auto"}'
```

**Request headers:**

| Header | Values | Default |
|---|---|---|
| `X-Enable-RAG` | `true` / `false` | `true` when RAG enabled globally |
| `X-Mode` | `normal` / `thinking` / `code` | `normal` |
| `X-Pipeline` | `chat` / `code` | `chat` |

---

## 🔌 API Endpoints

Once running, the API is available at `http://localhost:8000`.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the browser UI |
| `/health` | GET | Health check |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/v1/models` | GET | List available local models |
| `/v1/status` | GET | System status (hardware, VRAM, provider) |
| `/v1/cancel` | POST | Cancel in-progress generation |
| `/v1/providers` | GET | List registered inference providers |
| `/api/rag/documents/upload` | POST | Upload & ingest document (NotebookLM RAG) |
| `/api/rag/documents` | GET | List RAG documents with chunk counts |
| `/api/rag/documents/{id}` | DELETE | Delete document + chunks + physical files + vectors |
| `/api/rag/retrieve` | POST | Debug: raw chunk retrieval |
| `/api/rag/context` | POST | Debug: preview NotebookLM context |
| `/docs` | GET | Interactive API docs (Swagger) |
| `/v1/capabilities` | GET | Retrieve dynamic runtime capabilities status |
| `/v1/memory` | GET | Get working memory snapshot (variables, tasks, observations) |
| `/v1/memory/variables` | POST/DELETE | CRUD variables in working memory |
| `/v1/memory/tasks` | POST/PATCH/DELETE | CRUD tasks in working memory (status, priority, title) |
| `/v1/memory/observations` | POST/DELETE | CRUD observations with source provenance |
| `/v1/memory/reset` | POST | Clear all working memory for the session |
| `/api/documents/*` | * | *(Deprecated)* Legacy session-based document endpoints |

---

## 🧠 Runtime Capabilities

AS Core is hardware-aware, provider-aware, and runtime-modular. Functionalities are evaluated lazily using the **Runtime Capability System**. 

The UI queries `GET /v1/capabilities` to render controls dynamically rather than displaying inactive features.

Capabilities are organized by categories (`core`, `documents`, `tools`, `multimodal`, `developer`, `network`) and define explicit security `scopes` which active Skills consume. Users can explicitly enable/disable capabilities globally via the `capability_overrides` dictionary setting.

---

## 🔌 Compatibility & Connectors

AS Core exposes an OpenAI-compatible API, making it compatible with IDE tools and editors (like VSCode extensions Cline or Continue) as a custom backend. Configure your extension to use an OpenAI-compatible provider with the base URL pointing to `http://localhost:8000/v1` and any dummy API key.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details. Commercial use, modification, and redistribution are allowed. Attribution to Alpha Software is required.
