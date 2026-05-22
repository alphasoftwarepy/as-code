"""
AS Code — Working Memory API Routes

Runtime-native protocol. No OpenAI tool format — that comes later.

All endpoints scoped under /v1/memory.
session_id defaults to "default_session" when not provided.

The runtime-native payload format for capability execution (FASE 4):
    {"capability": "git", "action": "status", "params": {}}
is completely separate from these CRUD endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.database import get_db

logger = logging.getLogger("as-code.api.memory")

memory_router = APIRouter(prefix="/v1/memory", tags=["Working Memory"])

DEFAULT_SESSION = "default_session"


# ── Request schemas ────────────────────────────────────────────

class VariablePayload(BaseModel):
    session_id: str = Field(default=DEFAULT_SESSION)
    key: str = Field(..., min_length=1)
    value: str = Field(default="")


class TaskPayload(BaseModel):
    session_id: str = Field(default=DEFAULT_SESSION)
    title: str = Field(..., min_length=1)
    priority: int = Field(default=0, ge=0)


class TaskStatusPayload(BaseModel):
    status: str  # pending | in_progress | completed | failed


class ObservationPayload(BaseModel):
    session_id: str = Field(default=DEFAULT_SESSION)
    content: str = Field(..., min_length=1)
    source: str = Field(default="user")  # user | system | rag | capability


class ResetPayload(BaseModel):
    session_id: str = Field(default=DEFAULT_SESSION)


# ── Helper ─────────────────────────────────────────────────────

def _get_manager(request: Request):
    mem = getattr(request.app.state, "memory", None)
    if mem is None:
        raise HTTPException(
            status_code=503,
            detail="Working memory service not initialized"
        )
    return mem


# ── Snapshot ───────────────────────────────────────────────────

@memory_router.get("")
def get_memory(
    request: Request,
    session_id: str = DEFAULT_SESSION,
    db: Optional[Session] = Depends(get_db),
):
    """
    Full working memory snapshot for a session.
    Returns variables, tasks (ordered by priority desc), and observations.
    """
    if db is None:
        return {"session_id": session_id, "variables": [], "tasks": [], "observations": []}
    manager = _get_manager(request)
    return manager.get_memory(db, session_id)


# ── Variables ──────────────────────────────────────────────────

@memory_router.post("/variables", status_code=201)
def set_variable(
    payload: VariablePayload,
    request: Request,
    db: Optional[Session] = Depends(get_db),
):
    """Upsert a key-value variable for the session."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    result = manager.set_variable(db, payload.session_id, payload.key, payload.value)
    logger.info(f"[MEM-API] variable set — session={payload.session_id!r} key={payload.key!r}")
    return result


@memory_router.delete("/variables/{key}")
def delete_variable(
    key: str,
    request: Request,
    session_id: str = DEFAULT_SESSION,
    db: Optional[Session] = Depends(get_db),
):
    """Remove a variable by key."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    deleted = manager.delete_variable(db, session_id, key)
    if not deleted:
        raise HTTPException(404, f"Variable '{key}' not found in session '{session_id}'")
    logger.info(f"[MEM-API] variable deleted — session={session_id!r} key={key!r}")
    return {"deleted": True, "key": key}


# ── Tasks ──────────────────────────────────────────────────────

@memory_router.post("/tasks", status_code=201)
def add_task(
    payload: TaskPayload,
    request: Request,
    db: Optional[Session] = Depends(get_db),
):
    """Add a new task to the session task list."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    result = manager.add_task(
        db, payload.session_id, payload.title, payload.priority
    )
    logger.info(
        f"[MEM-API] task added — session={payload.session_id!r} "
        f"title={payload.title!r} priority={payload.priority}"
    )
    return result


@memory_router.patch("/tasks/{task_id}")
def update_task_status(
    task_id: str,
    payload: TaskStatusPayload,
    request: Request,
    db: Optional[Session] = Depends(get_db),
):
    """Update a task's status: pending | in_progress | completed | failed."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    try:
        result = manager.update_task_status(db, task_id, payload.status)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if result is None:
        raise HTTPException(404, f"Task '{task_id}' not found")
    logger.info(f"[MEM-API] task updated — id={task_id!r} status={payload.status!r}")
    return result


@memory_router.delete("/tasks/{task_id}")
def delete_task(
    task_id: str,
    request: Request,
    db: Optional[Session] = Depends(get_db),
):
    """Remove a task by ID."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    deleted = manager.delete_task(db, task_id)
    if not deleted:
        raise HTTPException(404, f"Task '{task_id}' not found")
    logger.info(f"[MEM-API] task deleted — id={task_id!r}")
    return {"deleted": True, "task_id": task_id}


# ── Observations ───────────────────────────────────────────────

@memory_router.post("/observations", status_code=201)
def add_observation(
    payload: ObservationPayload,
    request: Request,
    db: Optional[Session] = Depends(get_db),
):
    """Record an observation note with provenance source."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    try:
        result = manager.add_observation(
            db, payload.session_id, payload.content, payload.source
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    logger.info(
        f"[MEM-API] observation added — session={payload.session_id!r} "
        f"source={payload.source!r}"
    )
    return result


@memory_router.delete("/observations/{obs_id}")
def delete_observation(
    obs_id: str,
    request: Request,
    db: Optional[Session] = Depends(get_db),
):
    """Remove an observation by ID."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    deleted = manager.delete_observation(db, obs_id)
    if not deleted:
        raise HTTPException(404, f"Observation '{obs_id}' not found")
    logger.info(f"[MEM-API] observation deleted — id={obs_id!r}")
    return {"deleted": True, "obs_id": obs_id}


# ── Reset ──────────────────────────────────────────────────────

@memory_router.post("/reset")
def reset_memory(
    payload: ResetPayload,
    request: Request,
    db: Optional[Session] = Depends(get_db),
):
    """Clear all working memory for a session."""
    if db is None:
        raise HTTPException(503, "Database not initialized")
    manager = _get_manager(request)
    manager.reset(db, payload.session_id)
    logger.info(f"[MEM-API] session reset — session_id={payload.session_id!r}")
    return {"reset": True, "session_id": payload.session_id}
