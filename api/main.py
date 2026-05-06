"""
AS Code — FastAPI Application Entry Point

Ultra-lightweight async server serving:
- OpenAI-compatible API at /v1/
- Minimal web UI at /
- System health at /health

Startup sequence:
1. Detect hardware
2. Initialize provider registry
3. Register & activate inference provider
4. Register models
5. Start engine manager
6. Mount static UI files
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router
from config.settings import get_settings
from core.engine import EngineManager
from core.hardware import detect_hardware
from providers.litert_cli import LiteRTCLIProvider
from providers.litert_compiled import LiteRTCompiledProvider
from providers.registry import ProviderRegistry
from router.smart_router import SmartRouter

# ── Logging ────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("as-code")


# ── Application Lifespan ───────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("  AS Code — Local Edge AI Runtime")
    logger.info("  by Alpha Software")
    logger.info("=" * 60)

    # 1. Detect hardware
    hardware = detect_hardware()
    logger.info(f"Hardware: {hardware.summary()}")

    # 2. Create provider registry
    registry = ProviderRegistry()

    # 3. Register providers (all available backends)
    cli_provider = LiteRTCLIProvider(
        cli_path=settings.litert_cli_path,
        default_backend=settings.litert_backend,
        enable_speculative_decoding=settings.enable_speculative_decoding,
        models_dir=settings.models_dir,
    )
    registry.register("litert_cli", cli_provider)

    compiled_provider = LiteRTCompiledProvider(
        models_dir=settings.models_dir,
    )
    registry.register("litert_compiled", compiled_provider)

    # 4. Activate the configured provider
    await registry.set_active(settings.active_provider)

    # 5. Create engine manager
    engine = EngineManager(
        provider_registry=registry,
        hardware_info=hardware,
        max_vram_mb=settings.max_vram_usage_mb,
        model_unload_timeout=settings.model_unload_timeout_sec,
        anti_oom_threshold_mb=settings.anti_oom_threshold_mb,
    )

    # 6. Register models
    engine.register_model(
        model_id=settings.reasoning_model_id,
        model_path=settings.get_model_path(settings.reasoning_model_id),
        model_type="reasoning",
        estimated_vram_mb=settings.reasoning_model_vram_mb,
    )
    engine.register_model(
        model_id=settings.coding_model_id,
        model_path=settings.get_model_path(settings.coding_model_id),
        model_type="coding",
        estimated_vram_mb=settings.coding_model_vram_mb,
    )

    # 7. Create smart router
    smart_router = SmartRouter(
        reasoning_model=settings.reasoning_model_id,
        coding_model=settings.coding_model_id,
    )

    # 8. Start engine
    await engine.start()

    # Store in app state (accessible from routes)
    app.state.engine = engine
    app.state.router = smart_router
    app.state.settings = settings
    app.state.hardware = hardware

    logger.info(f"API ready at http://{settings.host}:{settings.port}/v1")
    logger.info(f"UI  ready at http://{settings.host}:{settings.port}/")
    logger.info(f"Active provider: {settings.active_provider}")
    logger.info(f"Models: {settings.reasoning_model_id}, {settings.coding_model_id}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down AS Code...")
    await engine.stop()
    logger.info("Goodbye!")


# ── FastAPI App ────────────────────────────────────────────────

app = FastAPI(
    title="AS Code",
    description="Ultra-optimized local AI inference for edge hardware",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,  # Disable redoc to save memory
)

# CORS (allow local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)


# ── UI Routes ──────────────────────────────────────────────────

# Serve static UI files
ui_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")
if os.path.exists(ui_dir):
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the minimal web UI."""
    index_path = os.path.join(ui_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        "<html><body><h1>AS Code</h1>"
        "<p>UI not found. Place index.html in /ui/</p>"
        '<p><a href="/docs">API Docs</a></p></body></html>'
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "as-code"}


# ── Direct Execution ───────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,  # No reload in production for minimal overhead
    )
