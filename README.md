# AS Code

**By Alpha Software**

AS Code is a local AI inference runtime designed for performance and stability on Windows. It provides a robust backend to run LLM models locally with full GPU acceleration, an OpenAI-compatible API, and SSE streaming.

> **Note:** This is an early open-source release focused on providing a stable runtime environment on Windows.

## Current Status

AS Code is currently in an early public release stage.

Core runtime functionality is already operational:
- LiteRT-LM Windows inference
- GPU acceleration
- OpenAI-compatible API
- SSE streaming
- local browser UI

Current focus:
- stability
- installation experience
- VSCode/Cline integration
- backend hardening

Advanced optimization and autonomous systems are planned for future phases.

## 🚀 Current Capabilities

* **LiteRT-LM Inference:** Running reliably on Windows.
* **GPU Acceleration:** Full support for local hardware acceleration.
* **FastAPI Backend:** High-performance serving framework.
* **OpenAI-Compatible API:** Drop-in replacement for OpenAI endpoints.
* **SSE Streaming:** Real-time token streaming support.
* **Local Web UI:** Minimal, functional browser-based UI included.
* **Hardware-Adaptive Runtime:** Automatically adjusts to your system.
* **Smart Routing & VRAM-Aware Loading:** Intelligent model management.
* **Hot Swap:** Base implemented for fast model switching.

## 📸 Screenshots

*(Placeholders for screenshots)*
- `[Screenshot of Local Web UI]`
- `[Screenshot of Terminal Output]`
- `[Screenshot of VSCode Integration]`

## 💻 Hardware Target

- **OS:** Windows 10/11
- **GPU:** Dedicated GPU highly recommended for optimal performance

## 🏗 Architecture Summary

AS Code uses a modular architecture built on top of FastAPI and LiteRT-LM. It acts as an intelligent routing and execution layer for local models, abstracting away the complexity of VRAM management and hardware-specific configurations, while exposing a standard OpenAI-compatible REST API.

## 🛠 Installation

### Prerequisites
- Windows 10/11
- PowerShell
- Python 3.10+
- (Optional but recommended) Compatible GPU drivers

### Setup

Clone the repository and run the setup script:

```powershell
git clone https://github.com/cursosdigitaleshd-del/as-code.git
cd as-code
.\scripts\install.ps1
```

## 🧠 Manual Model Setup (Important)

> **Important Limitation:** Currently, models are **NOT automatically downloaded** due to HuggingFace authentication and token requirements. Automatic model management will be added in a future release. 

For now, you must manually download your models and place them in the correct directory:

1. Download the `.litertlm` model file manually from your official source (e.g., HuggingFace).
2. Place the downloaded `.litertlm` file inside the `models/` directory in this project.
3. The setup script and runtime will automatically detect and register models placed in this folder.

## 🏃‍♂️ Running the Project

Start the local server using the provided script:

```powershell
.\scripts\run.ps1
```

This will activate the environment, start the FastAPI server, and output logs cleanly.

## 🔌 API Endpoints

Once running, the API is available at `http://localhost:8000`.

- `GET /` - Health check
- `POST /v1/chat/completions` - OpenAI-compatible chat completions endpoint (supports `stream=true`)
- `GET /v1/models` - List available local models

## 🔌 VSCode / Cline Compatibility

AS Code exposes an OpenAI-compatible API, making it fully compatible with VSCode extensions like Cline. Simply configure your extension to use an OpenAI-compatible provider with the base URL pointing to `http://localhost:8000/v1` and any dummy API key.

## 🗺 Roadmap Overview

We are building incrementally. See [ROADMAP.md](ROADMAP.md) for full details.

- **Completed Phases:** Core architecture, LiteRT Windows runtime, GPU support, OpenAI API, minimal UI.
- **Upcoming Phases:** Hardening, logging, error handling, queueing.
- **Future Goals:** Multi-model execution, Agents, Workflows, Autonomous systems, Advanced memory optimization, Automatic model downloads.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 💖 Support the Project

If AS Code helps you, consider supporting development.

Your support helps improve:

* LiteRT Windows optimization
* local AI infrastructure
* VSCode/Cline integration
* performance optimization
* future autonomous systems

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
