"""
AS Code — Working Memory Models

Three lightweight tables for short-term agent state:
  memory_variables    — key-value workspace context
  memory_tasks        — ordered task list with priority + status
  memory_observations — agent/user notes with provenance source

ALL tables carry session_id for future multi-chat / multi-project isolation.
Default session: "default_session" (used until auth/project system is added).

Design rules:
  - NO automatic LLM extraction — manual writes only (Phase 2 is infrastructure)
  - session_id is indexed everywhere — future-proof for per-tab, per-user isolation
  - priority on tasks — Main Agent will use this in FASE 3
  - source on observations — critical for debuggability and future orchestration
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class MemoryBase(DeclarativeBase):
    """Separate declarative base to keep memory tables isolated from RAG base."""
    pass


class MemoryVariable(MemoryBase):
    """
    Key-value context variable scoped to a session.

    Examples:
      key="workspace_root"  value="/home/user/myproject"
      key="current_file"    value="api/routes.py"
      key="active_branch"   value="feat/memory-layer"
    """
    __tablename__ = "memory_variables"

    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id = Column(
        String, index=True, nullable=False, default="default_session"
    )
    key = Column(String, nullable=False, index=True)
    value = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MemoryTask(MemoryBase):
    """
    Agent task with status and priority, scoped to a session.

    Status lifecycle: pending → in_progress → completed | failed

    Priority: integer, higher = more important.
    (FASE 3 Main Agent will schedule by priority desc, then order asc)
    """
    __tablename__ = "memory_tasks"

    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id = Column(
        String, index=True, nullable=False, default="default_session"
    )
    title = Column(String, nullable=False)
    status = Column(
        String, nullable=False, default="pending"
        # pending | in_progress | completed | failed
    )
    priority = Column(
        Integer, nullable=False, default=0
        # 0 = low, higher int = higher priority
        # Main Agent reads this in FASE 3
    )
    order = Column(
        Integer, nullable=False, default=0
        # display order within same priority tier
    )
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    STATUS_ICON = {
        "pending": "[ ]",
        "in_progress": "[/]",
        "completed": "[x]",
        "failed": "[!]",
    }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "order": self.order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MemoryObservation(MemoryBase):
    """
    A recorded fact or note, scoped to a session.

    source values:
      "user"       — written directly by the human operator
      "system"     — written by runtime startup/events
      "rag"        — derived from a retrieval result
      "capability" — produced by a capability execution (FASE 4+)

    Source is indexed for future filtering/debugging.
    """
    __tablename__ = "memory_observations"

    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id = Column(
        String, index=True, nullable=False, default="default_session"
    )
    content = Column(Text, nullable=False)
    source = Column(
        String, nullable=False, default="user", index=True
        # user | system | rag | capability
    )
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }
