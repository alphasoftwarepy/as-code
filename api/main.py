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
from api.document_routes import documents_router
from api.rag_routes import rag_router
from api.capability_routes import capabilities_router
from api.skill_routes import skills_router
from api.memory_routes import memory_router
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
    logger.info("  AS Code — Fast Local AI Runtime")
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
    for role, cfg in settings.models.items():
        engine.register_model(
            model_id=role,
            model_path=settings.get_model_path(role),
            model_type=cfg.get("type", "general"),
            estimated_vram_mb=cfg.get("estimated_vram_mb", 1500),
            provider_id=cfg.get("provider"),
        )

    # 7. Create smart router
    smart_router = SmartRouter(
        chat_model="chat",
        coding_model="code",
        reasoning_model="reasoning",
    )

    # 8. Start engine
    await engine.start()

    # Create uploads directory for document sessions
    from pathlib import Path
    Path(settings.uploads_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(settings.uploads_dir) / "rag").mkdir(parents=True, exist_ok=True)

    # ── RAG NotebookLM Pipeline ────────────────────────────────
    if settings.enable_rag_mode:
        logger.info("Initializing RAG NotebookLM pipeline...")
        try:
            from api.database import init_db
            from api.rag_service import build_rag_service

            init_db(settings.rag_db_path)

            rag_service = build_rag_service(
                faiss_index_path=settings.faiss_index_path,
                embedding_dim=settings.rag_embedding_dim,
                embedder_model=settings.rag_embedder_model,
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
                retrieval_mode=settings.rag_retrieval_mode,
                hybrid_alpha=settings.rag_hybrid_alpha,
            )
            app.state.rag_service = rag_service
            logger.info("RAG pipeline ready ✓")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e} — continuing without RAG")
            app.state.rag_service = None
    else:
        app.state.rag_service = None
        logger.info("RAG mode disabled (set ASCODE_ENABLE_RAG_MODE=true to enable)")

    # Load skills at startup
    from runtime.skills.loader import get_skill_loader
    skill_loader = get_skill_loader()
    skill_loader.load_skills()
    app.state.skill_service = skill_loader

    # ── Working Memory ─────────────────────────────────────────
    from runtime.memory.manager import WorkingMemoryManager
    app.state.memory = WorkingMemoryManager()
    logger.info("Working memory service ready ✓")

    # Store in app state (accessible from routes)
    app.state.engine = engine
    app.state.router = smart_router
    app.state.settings = settings
    app.state.hardware = hardware

    logger.info(f"API ready at http://{settings.host}:{settings.port}/v1")
    logger.info(f"UI  ready at http://{settings.host}:{settings.port}/")
    logger.info(f"Active provider: {settings.active_provider}")
    logger.info(f"Models: {list(settings.models.keys())}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down AS Code...")
    await engine.stop()
    logger.info("Goodbye!")


# ── FastAPI App ────────────────────────────────────────────────

app = FastAPI(
    title="AS Code",
    description="Fast, lightweight, general-purpose local AI runtime for modest hardware",
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

# Prevent caching of static assets (JS/CSS)
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount API routes
app.include_router(api_router)
app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(capabilities_router)
app.include_router(skills_router)
logger.info("Skills router mounted at /v1/skills ✓")
app.include_router(memory_router)
logger.info("Memory router mounted at /v1/memory ✓")


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
        response = FileResponse(index_path)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
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
