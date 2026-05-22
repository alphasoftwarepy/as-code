"""
AS Code — API Routes

OpenAI-compatible endpoints:
- POST /v1/chat/completions — Chat inference (streaming + non-streaming)
- GET  /v1/models           — List available models
- GET  /v1/status           — System status (AS Code extension)
- POST /v1/cancel           — Cancel generation (AS Code extension)
- GET  /v1/providers        — List providers (AS Code extension)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelInfo,
    ModelListResponse,
    StatusResponse,
    UsageInfo,
)
from api.streaming import stream_inference_results
from api.document_service import get_document_service
from api.database import get_db
from providers.base import InferenceRequest

logger = logging.getLogger("as-code.api.routes")

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


# ── POST /v1/chat/completions ──────────────────────────────────


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """OpenAI-compatible chat completion endpoint.

    Supports:
    - Streaming (SSE) and non-streaming responses
    - Smart routing via model="auto"
    - Explicit model selection
    - Temperature and token control
    - RAG NotebookLM context injection (X-Enable-RAG / X-Mode / X-Pipeline headers)
    """
    engine = request.app.state.engine
    smart_router = request.app.state.router
    settings = request.app.state.settings

    # Generate request ID for tracking and cancellation
    request_id = body.get_request_id()

    # Get the last user message for routing
    user_message = body.get_last_user_message()
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message provided")

    # Route to optimal model first to align RAG settings
    model_param = body.model if body.model != "auto" else None
    model_id, system_prompt = smart_router.route(user_message, model_param)

    # ── Skill Runtime: inject skill prompt ─────────────────────
    skill_id = request.headers.get("X-Skill")
    if skill_id:
        skill_service = getattr(request.app.state, "skill_service", None)
        if skill_service:
            skill_prompt = skill_service.get_skill_prompt(skill_id)
            if skill_prompt:
                system_prompt = f"{system_prompt}\n\n{skill_prompt}"
                logger.info(
                    f"[SKILL-INJECT] skill={skill_id!r} | "
                    f"prompt_chars={len(skill_prompt)} | "
                    f"preview={skill_prompt[:60].replace(chr(10), ' ')!r}"
                )

    # ── Working Memory: inject cognitive state into system prompt ──
    # Injection order: base_prompt → skill_prompt → [Working Memory]
    # RAG context goes into the user message — Working Memory stays in SYSTEM.
    memory_service = getattr(request.app.state, "memory", None)
    if memory_service:
        session_id = request.headers.get("X-Session-Id", "default_session")
        try:
            memory_block = memory_service.format_prompt_block(db, session_id)
            if memory_block:
                system_prompt = f"{system_prompt}\n\n{memory_block}"
                logger.info(
                    f"[MEM-INJECT] session={session_id!r} | "
                    f"block_chars={len(memory_block)}"
                )
        except Exception as _mem_err:
            # Graceful degradation — memory failure never blocks inference
            logger.warning(f"Working memory inject failed (degrading): {_mem_err}")

    # ── RAG NotebookLM: inject context ─────────────────────────
    # Header: X-Enable-RAG: true/false  (default: true when RAG enabled globally)
    # Header: X-Mode: normal | thinking | code
    # Header: X-Pipeline: chat | code
    rag_was_used = False
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service and settings.enable_rag_mode:
        enable_rag = request.headers.get("X-Enable-RAG", "true").lower() != "false"
        if enable_rag and body.messages:
            # Dynamically set mode and pipeline based on routing if not explicitly requested via headers
            header_mode = request.headers.get("X-Mode")
            header_pipeline = request.headers.get("X-Pipeline")
            
            if header_mode:
                mode = header_mode
            else:
                mode = "thinking" if model_id == "reasoning" else ("code" if model_id == "code" else "normal")
                
            if header_pipeline:
                pipeline = header_pipeline
            else:
                pipeline = "code" if model_id == "code" else "chat"

            try:
                context = rag_service.build_context(
                    query=user_message,
                    db=db,
                    mode=mode,
                    pipeline=pipeline,
                )
                if context:
                    last_msg = body.messages[-1]
                    last_msg.content = (
                        f"{context}\n\n---USER QUESTION---\n{last_msg.content}"
                    )
                    rag_was_used = True
                    logger.info(
                        f"[RAG-INJECT] mode={mode} | pipeline={pipeline} | "
                        f"context_chars={len(context)} | "
                        f"preview={context[:80].replace(chr(10),' ')!r}"
                    )
                else:
                    logger.info(
                        f"[RAG-RETRIEVE-EMPTY] mode={mode} | pipeline={pipeline} | "
                        f"no chunks found — FAISS may be empty (upload a document first)"
                    )
            except Exception as _rag_err:
                # Graceful degradation: log and continue without RAG
                logger.warning(f"RAG context failed (degrading gracefully): {_rag_err}")

    # ── Legacy session-based document injection (kept for backward compat) ─
    # ONLY activates when:
    #   (a) RAG mode is globally disabled (ASCODE_ENABLE_RAG_MODE=false), AND
    #   (b) A session ID header is present
    # Will NOT fire when RAG is enabled — even if FAISS is empty.
    session_id = request.headers.get("X-Document-Session-Id")
    if session_id and not settings.enable_rag_mode:
        doc_context = get_document_service().get_context(session_id, max_chars=8000)
        if doc_context and body.messages:
            last = body.messages[-1]
            last.content = f"{doc_context}\n\n---PREGUNTA---\n{last.content}"
            logger.info(f"[LEGACY-INJECT] session_id={session_id!r} | chars={len(doc_context)}")

    # Build the inference request (provider-agnostic)
    inference_request = InferenceRequest(
        prompt=body.build_prompt(),
        model_id=model_id,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p,
        top_k=body.top_k,
        stop_sequences=body.stop or [],
        stream=body.stream,
        system_prompt=system_prompt,
        request_id=request_id,
    )

    if body.stream:
        # Streaming response (SSE)
        result_stream = engine.generate_stream(inference_request)

        return StreamingResponse(
            stream_inference_results(result_stream, model_id, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )
    else:
        # Non-streaming response
        start = time.perf_counter()
        result = await engine.generate(inference_request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason=result.finish_reason or "stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.tokens_generated,
                total_tokens=result.prompt_tokens + result.tokens_generated,
            ),
            provider=result.provider_type,
            tokens_per_sec=result.tokens_per_sec,
            latency_ms=elapsed_ms,
        )


# ── GET /v1/models ─────────────────────────────────────────────


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request):
    """List available models."""
    engine = request.app.state.engine
    models = engine.get_registered_models()

    return ModelListResponse(
        data=[
            ModelInfo(
                id=m["id"],
                owned_by=m.get("owned_by", "as-code"),
            )
            for m in models
        ]
    )


# ── GET /v1/status ─────────────────────────────────────────────


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request):
    """Get system status including hardware, models, and provider info."""
    engine = request.app.state.engine
    status = await engine.get_status()
    return StatusResponse(**status)


# ── POST /v1/cancel ────────────────────────────────────────────


@router.post("/cancel")
async def cancel_generation(request: Request, request_id: str = "", model_id: str = ""):
    """Cancel an in-progress generation."""
    if not request_id:
        raise HTTPException(status_code=400, detail="request_id required")

    engine = request.app.state.engine
    # We don't have model_id here usually, but engine.cancel_generation 
    # will fall back to active_provider if not provided.
    # Pass model_id to engine so it can route to the correct provider
    await engine.cancel_generation(request_id, model_id)
    return {"status": "cancelled", "request_id": request_id, "model_id": model_id}


# ── GET /v1/providers ──────────────────────────────────────────


@router.get("/providers")
async def list_providers(request: Request):
    """List registered inference providers and their status."""
    engine = request.app.state.engine
    return engine.registry.list_providers()
