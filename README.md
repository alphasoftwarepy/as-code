# AS Code

**By Alpha Software**

AS Code is a lightweight, general-purpose local AI runtime designed for speed and simplicity on modest hardware. It provides a robust, Windows-optimized environment to run large language models locally with minimal overhead, a browser-first chat experience, and an OpenAI-compatible API.

> **AS Code is NOT just for coding.** It is a personal AI platform for ideas, productivity, planning, writing, and local experimentation.

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

## 🚀 Key Features

*   **LiteRT-LM Runtime:** Ultra-optimized inference engine for Windows.
*   **Hardware-Adaptive:** Automatically adjusts to your system's VRAM and CPU capabilities.
*   **Browser-First Experience:** A premium, minimal local web UI for everyday use.
*   **OpenAI-Compatible API:** Use AS Code as a drop-in backend for any tool that supports OpenAI.
*   **Low VRAM Optimization:** Intelligent model loading and hot-swapping.
*   **Optional Extensions:** Seamlessly integrates with VSCode/Cline when you need a coding companion.

## 📸 Experience AS Code

*(Screenshots coming soon)*
- **Personal Assistant:** Drafting emails and planning workflows.
- **Creative Hub:** Writing stories and brainstorming ideas.
- **Local Server:** Powering third-party AI tools via the API.

## 📸 Screenshots

*(Placeholders for screenshots)*
- `[Screenshot of Local Web UI]`
- `[Screenshot of Terminal Output]`
- `[Screenshot of VSCode Integration]`

## 💻 Hardware Philosophy

AS Code is built for "Real Hardware"—the laptops and desktops people actually own. While a dedicated GPU is recommended for the best experience, our architecture is designed to remain responsive even on mid-range systems.

- **Optimized for:** Windows 10/11
- **Focus:** Maximum performance per watt/GB.

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

> **Important Limitation:** Models are **NOT automatically downloaded** due to HuggingFace authentication requirements. Automatic model management is planned for a future release.

AS Code uses a dual-model architecture with two GPU-accelerated Gemma LiteRT models:

| Role           | Model ID        | File path                                          |
|----------------|-----------------|----------------------------------------------------|
| General / Chat | `gemma-3n-web`  | `models/gemma/gemma-3n-E2B-it-int4-Web.litertlm`  |
| Coding         | `gemma-3n-code` | `models/gemma/gemma-3n-E2B-it-int4.litertlm`      |

**Setup steps:**

1. Create the directory: `models\gemma\`
2. Download both `.litertlm` files from [HuggingFace — litert-community](https://huggingface.co/litert-community).
3. Place them in the paths shown above.
4. Run the server — the runtime detects and registers them automatically.


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
