# AS Code — Models Directory

Place your LiteRT model files here before running the server.

## Required Models

AS Code uses a dual-model architecture powered by GPU-accelerated Gemma LiteRT models:

| Role           | Model ID        | File path                                     |
|----------------|-----------------|-----------------------------------------------|
| General / Chat | `gemma-3n-web`  | `models/gemma/gemma-3n-E2B-it-int4.litertlm` |
| Coding         | `gemma-3n-code` | `models/gemma/gemma-3n-E2B-it-int4.litertlm` |

> Both roles use the same model file. Routing is handled via system prompt differentiation.

## Setup

1. Create the `models/gemma/` directory.
2. Download the `.litertlm` files manually from HuggingFace.
3. Place them in the correct paths shown above.
4. Run `.\scripts\run.ps1` — the runtime will auto-detect them.

## Sources

- https://huggingface.co/litert-community/gemma-3n-E2B-it-litert-lm
