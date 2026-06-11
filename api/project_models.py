"""
AS Code — Project Models

Entities:
  projects         — isolated work contexts
  project_chats    — conversations (chats) mapped to a project
  project_documents — RAG documents mapped to a project (composite PK to enforce uniqueness)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase


class ProjectBase(DeclarativeBase):
    """Declarative base for project management models."""
    pass


class Project(ProjectBase):
    """
    Project entity representing an isolated workspace.
    """
    __tablename__ = "projects"

    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ProjectChat(ProjectBase):
    """
    Chat session belonging to a project.
    Note: session_id is decoupled from the chat's internal id.
    """
    __tablename__ = "project_chats"

    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id = Column(
        String, unique=True, index=True, nullable=False
    )
    project_id = Column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title = Column(String, nullable=False, default="Nuevo Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ProjectDocument(ProjectBase):
    """
    Mapping of a RAG document to a project.
    Composite primary key ensures unique (project_id, document_id) association.
    """
    __tablename__ = "project_documents"

    project_id = Column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    document_id = Column(
        String, primary_key=True  # Reference to rag_documents.id
    )
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "document_id": self.document_id,
            "created_at": self.created_at.isoformat(),
        }


class ProjectChatMessage(ProjectBase):
    """
    Message history of a chat session.
    """
    __tablename__ = "project_chat_messages"

    id = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id = Column(
        String, index=True, nullable=False
    )
    role = Column(String, nullable=False)  # 'user' | 'assistant' | 'system' | 'tool'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }

