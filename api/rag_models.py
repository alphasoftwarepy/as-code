"""
AS Code — RAG SQLAlchemy Models

SQLite stores metadata ONLY. No embedding column — embeddings live in FAISS.

Tables:
  rag_documents      — uploaded files
  rag_document_chunks — chunks with symbolic metadata (symbol, file, section)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class RAGDocument(Base):
    """Uploaded document — full text + metadata."""

    __tablename__ = "rag_documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, index=True, nullable=False)
    file_type = Column(String, nullable=False)   # py|txt|md|pdf|docx
    content = Column(Text, nullable=False)        # Full extracted text
    source = Column(String, default="local")      # local|web
    pipeline = Column(String, default="chat")     # chat|code
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    chunks = relationship(
        "RAGDocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        chunk_count = len(self.chunks) if self.chunks else 0
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "source": self.source,
            "pipeline": self.pipeline,
            "created_at": self.created_at.isoformat(),
            "chunk_count": chunk_count,
            # Derived ingest state — no extra DB column needed:
            # "indexing" = doc saved but background chunking not yet done
            # "ready"    = at least one chunk exists in SQLite
            "ingest_status": "ready" if chunk_count > 0 else "indexing",
        }


class RAGDocumentChunk(Base):
    """
    Document chunk for RAG retrieval.

    Note: NO embedding column. Embeddings are stored in FAISS.
    The chunk id is used as the bridge between SQLite and FAISS.

    meta_json stores symbolic metadata for VS Code / Code Graph integration:
      {"symbol": "register_model", "symbol_type": "function",
       "file": "engine.py", "section": "Model Registration",
       "line_start": 42, "line_end": 78, "language": "python"}
    """

    __tablename__ = "rag_document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("rag_documents.id"), index=True)
    section_name = Column(String)
    chunk_index = Column(Integer)
    chunk_type = Column(String, default="text")  # text|code|markdown_section
    text = Column(Text, nullable=False)
    meta_json = Column(Text, default="{}")        # JSON symbolic metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("RAGDocument", back_populates="chunks")

    # ── metadata property ──────────────────────────────────────

    @property
    def meta(self) -> dict:
        try:
            return json.loads(self.meta_json or "{}")
        except Exception:
            return {}

    @meta.setter
    def meta(self, value: dict) -> None:
        self.meta_json = json.dumps(value)

    def to_dict(self) -> dict:
        text_preview = self.text[:200] + "..." if len(self.text) > 200 else self.text
        return {
            "id": self.id,
            "document_id": self.document_id,
            "section": self.section_name,
            "index": self.chunk_index,
            "type": self.chunk_type,
            "text": text_preview,
            "metadata": self.meta,
        }
