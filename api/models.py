"""
AS Code — Pydantic Request/Response Models

OpenAI-compatible API schemas for /v1/chat/completions and /v1/models.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(
        default="auto",
        description="Model ID or 'auto' for smart routing",
    )
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Conversation messages",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    stream: bool = Field(default=False)
    stop: Optional[list[str]] = Field(default=None)

    # Extension: request tracking
    request_id: Optional[str] = None

    def get_request_id(self) -> str:
        return self.request_id or f"req_{uuid.uuid4().hex[:12]}"

    def get_last_user_message(self) -> str:
        """Extract the last user message for routing."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""

    def build_prompt(self) -> str:
        """Build a single prompt string from messages.
        Used by CLI-based providers that don't support multi-turn natively."""
        parts = []
        for msg in self.messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)


# ── Response Models ────────────────────────────────────────────


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int = 0
    message: ChatMessage = Field(default_factory=lambda: ChatMessage(role="assistant"))
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)

    # AS Code extensions
    provider: Optional[str] = None
    tokens_per_sec: Optional[float] = None
    latency_ms: Optional[float] = None


# ── Streaming Models ───────────────────────────────────────────


class DeltaContent(BaseModel):
    """Delta content for streaming responses."""
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """A streaming choice."""
    index: int = 0
    delta: DeltaContent = Field(default_factory=DeltaContent)
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice] = Field(default_factory=list)


# ── Model List Models ─────────────────────────────────────────


class ModelInfo(BaseModel):
    """Model information for /v1/models."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "as-code"


class ModelListResponse(BaseModel):
    """Response for GET /v1/models."""
    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)


# ── Status Models ──────────────────────────────────────────────


class StatusResponse(BaseModel):
    """Response for GET /v1/status (AS Code extension)."""
    active_model: Optional[str] = None
    hardware_tier: str = "unknown"
    ram_available_mb: int = 0
    gpu: dict[str, Any] = Field(default_factory=dict)
    provider: dict[str, Any] = Field(default_factory=dict)
    registered_models: list[str] = Field(default_factory=list)
