# AS Code

**By Alpha Software**

AS Code is a lightweight, general-purpose local AI runtime designed for speed and simplicity on modest hardware. It provides a robust, Windows-optimized environment to run large language models locally with minimal overhead, a browser-first chat experience, and an OpenAI-compatible API.

> **AS Code is NOT just for coding.** It is a personal AI platform for ideas, productivity, planning, writing, local experimentation — and now, document-aware conversations.

## Current Status

AS Code is currently in an early public release stage.

Core runtime functionality is already operational:
- LiteRT-LM Windows inference
- GPU acceleration
- OpenAI-compatible API
- SSE streaming
- Local browser UI
- **Document upload & RAG context injection** (TXT / PDF / DOCX)

Current focus:
- Stability
- Installation experience
- VSCode/Cline integration
- Backend hardening

Advanced optimization and autonomous systems are planned for future phases.

## 🚀 Key Features

*   **LiteRT-LM Runtime:** Ultra-optimized inference engine for Windows.
*   **Hardware-Adaptive:** Automatically adjusts to your system's VRAM and CPU capabilities.
*   **Browser-First Experience:** A premium, minimal local web UI for everyday use.
*   **OpenAI-Compatible API:** Use AS Code as a drop-in backend for any tool that supports OpenAI.
*   **Low VRAM Optimization:** Intelligent model loading and hot-swapping.
*   **Document-Aware Chat:** Upload TXT, PDF, or DOCX files and chat with their contents directly in the browser.
*   **Optional Extensions:** Seamlessly integrates with VSCode/Cline when you need a coding companion.

## 📸 Screenshots

### Local AI Chat

![AS Code UI](screenshots/ui-chat.png)

### Features shown
- Multi-model routing
- GPU acceleration
- Local inference
- Real-time streaming
- Browser-based UI
- Document upload panel

## 💻 Hardware Philosophy

AS Code is built for "Real Hardware"—the laptops and desktops people actually own. While a dedicated GPU is recommended for the best experience, our architecture is designed to remain responsive even on mid-range systems.

- **Optimized for:** Windows 10/11
- **Focus:** Maximum performance per watt/GB.

## 🏗 Architecture Summary

AS Code uses a modular architecture built on top of FastAPI and LiteRT-LM. It acts as an intelligent routing and execution layer for local models, abstracting away the complexity of VRAM management and hardware-specific configurations, while exposing a standard OpenAI-compatible REST API.

Document context injection is handled as a thin layer between the API and the engine — zero changes to inference infrastructure.

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
| `psutil` | System monitoring |
| `pyyaml` | Config file parsing |
| `httpx` | HTTP client |

To install manually:
```powershell
pip install -r requirements.txt
```

## 🧠 Manual Model Setup (Important)

AS Code uses a **Role-Based Architecture**. The internal logic doesn't care about specific model names, only about the role the model plays.

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

## 🏃‍♂️ Running the Project

Start the local server using the provided script:

```powershell
.\scripts\run.ps1
```

This will activate the environment, start the FastAPI server, and output logs cleanly.

Once running, open your browser at `http://localhost:8000`.

## 📄 Document Upload (RAG)

AS Code supports uploading documents to chat with their contents. This works entirely locally — no cloud involved.

### Supported formats
- **TXT** — plain text files
- **PDF** — text-based PDFs (not scanned images)
- **DOCX** — Microsoft Word documents

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
| `/api/documents/session` | POST | Create document session |
| `/api/documents/upload` | POST | Upload TXT/PDF/DOCX |
| `/api/documents/{id}` | GET | List documents in session |
| `/api/documents/{id}` | DELETE | Clear document session |
| `/docs` | GET | Interactive API docs (Swagger) |

## 🔌 VSCode / Cline Compatibility

AS Code exposes an OpenAI-compatible API, making it fully compatible with VSCode extensions like Cline. Simply configure your extension to use an OpenAI-compatible provider with the base URL pointing to `http://localhost:8000/v1` and any dummy API key.

## 🗺 Roadmap Overview

- **Completed:** Core architecture, LiteRT Windows runtime, GPU support, OpenAI API, minimal UI, document upload & RAG context injection.
- **Upcoming:** VSCode/Cline integration guide, install hardening, GPU fallback, YAML-driven multi-model profiles.
- **Future Goals:** Agents, Workflows, Autonomous systems, Advanced memory optimization, Automatic model downloads, vector-store RAG.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 💖 Support the Project

If AS Code helps you, consider supporting development.

Your support helps improve:

* LiteRT Windows optimization
* Local AI infrastructure
* VSCode/Cline integration
* Performance optimization
* RAG and document intelligence
* Future autonomous systems

### Crypto Donations

USDT (TRC20)

```text
TADArdWELAAQMVtufWzcfF3R2yNPnyRfXr
```

IMPORTANT:
Please send only USDT using the TRON (TRC20) network.

Thank you for supporting open local AI infrastructure development.

— AS Code / Alpha Software

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details. Commercial use, modification, and redistribution are allowed. Attribution to Alpha Software is required.
