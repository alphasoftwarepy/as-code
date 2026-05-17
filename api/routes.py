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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

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
from providers.base import InferenceRequest

logger = logging.getLogger("as-code.api.routes")

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


# ── POST /v1/chat/completions ──────────────────────────────────


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completion endpoint.

    Supports:
    - Streaming (SSE) and non-streaming responses
    - Smart routing via model="auto"
    - Explicit model selection
    - Temperature and token control
    """
    engine = request.app.state.engine
    smart_router = request.app.state.router

    # Generate request ID for tracking and cancellation
    request_id = body.get_request_id()

    # Get the last user message for routing
    user_message = body.get_last_user_message()
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message provided")

    # ── RAG: Inyectar contexto de documentos si hay sesión ──────
    session_id = request.headers.get("X-Document-Session-Id")
    if session_id:
        doc_context = get_document_service().get_context(session_id, max_chars=8000)
        if doc_context and body.messages:
            last = body.messages[-1]
            last.content = f"{doc_context}\n\n---PREGUNTA---\n{last.content}"

    # Route to optimal model
    model_param = body.model if body.model != "auto" else None
    model_id, system_prompt = smart_router.route(user_message, model_param)

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
