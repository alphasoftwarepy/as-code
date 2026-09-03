"""
AS Code — Graph SQLAlchemy Models

Three tables for the optional Graph layer:
  graph_nodes         — named entities extracted from project documents
  graph_edges         — directed relationships between entities
  graph_build_status  — per-project build state (not_built | entities_only | full)

Design rules:
  - project_id is mandatory and indexed on EVERY table (isolation invariant).
  - document_id is provenance only — content is NOT stored here (RAG owns text).
  - JSON columns (aliases_json, metadata_json) are lightweight bags, never full docs.
  - session_id is NOT an ownership field for Graph data.
  - All primary keys are UUIDs (strings) consistent with rag_models.py / memory_models.py.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase


class GraphBase(DeclarativeBase):
    """Separate declarative base keeps Graph tables isolated from RAG, Memory and Project bases."""
    pass


class GraphNode(GraphBase):
    """
    A named entity belonging to a project's knowledge graph.

    Ownership: project_id.
    Provenance: source_doc_ids_json (list of document IDs where the entity appears).
    Content: NOT stored here — only structural and semantic metadata.

    Idempotency key: (project_id, label_normalized) — enforced via UniqueConstraint.
    Two nodes with the same normalized label in the same project are the same entity.
    Across projects they are independent records.
    """

    __tablename__ = "graph_nodes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, index=True, nullable=False)
    label = Column(String, nullable=False)
    label_normalized = Column(String, nullable=False, index=True)   # lowercase, stripped
    entity_type = Column(String, nullable=False, default="concept")  # person|org|place|date|concept|file|contract|other
    aliases_json = Column(Text, default="[]")                        # JSON list of surface forms
    source_doc_ids_json = Column(Text, default="[]")                 # JSON list of document_id strings (provenance)
    metadata_json = Column(Text, default="{}")                       # JSON key-value extras (domain hints, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("project_id", "label_normalized", name="uq_graph_node_project_label"),
    )

    # ── JSON accessors ──────────────────────────────────────────

    @property
    def aliases(self) -> list[str]:
        try:
            return json.loads(self.aliases_json or "[]")
        except Exception:
            return []

    @aliases.setter
    def aliases(self, value: list[str]) -> None:
        self.aliases_json = json.dumps(value)

    @property
    def source_doc_ids(self) -> list[str]:
        try:
            return json.loads(self.source_doc_ids_json or "[]")
        except Exception:
            return []

    @source_doc_ids.setter
    def source_doc_ids(self, value: list[str]) -> None:
        self.source_doc_ids_json = json.dumps(value)

    @property
    def meta(self) -> dict:
        try:
            return json.loads(self.metadata_json or "{}")
        except Exception:
            return {}

    @meta.setter
    def meta(self, value: dict) -> None:
        self.metadata_json = json.dumps(value)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "label": self.label,
            "label_normalized": self.label_normalized,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "source_doc_ids": self.source_doc_ids,
            "metadata": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class GraphEdge(GraphBase):
    """
    A directed relationship between two GraphNodes in the same project.

    Ownership: project_id (enforced — both nodes must belong to the same project).
    Provenance: source_doc_id (the document where this relationship was detected).

    Idempotency key: (project_id, source_node_id, target_node_id, relation_type, source_doc_id)
    — enforced via UniqueConstraint to prevent duplicate edge insertions for
    the same document passage.
    """

    __tablename__ = "graph_edges"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String, index=True, nullable=False)
    source_node_id = Column(String, index=True, nullable=False)  # GraphNode.id
    target_node_id = Column(String, index=True, nullable=False)  # GraphNode.id
    relation_type = Column(String, nullable=False)               # e.g. "firma", "importa", "propietario"
    source_doc_id = Column(String, nullable=True, index=True)    # Provenance document_id (optional)
    confidence = Column(Float, default=1.0)                      # [0.0, 1.0]
    metadata_json = Column(Text, default="{}")                   # Lightweight extras
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "project_id", "source_node_id", "target_node_id", "relation_type", "source_doc_id",
            name="uq_graph_edge_project_src_tgt_rel_doc",
        ),
    )

    @property
    def meta(self) -> dict:
        try:
            return json.loads(self.metadata_json or "{}")
        except Exception:
            return {}

    @meta.setter
    def meta(self, value: dict) -> None:
        self.metadata_json = json.dumps(value)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relation_type": self.relation_type,
            "source_doc_id": self.source_doc_id,
            "confidence": self.confidence,
            "metadata": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class GraphBuildStatus(GraphBase):
    """
    Per-project graph construction state.

    One row per project. Tracks whether entity extraction and/or edge building
    has been done for this project's document corpus.

    status values:
      not_built      — no graph data exists for this project
      entities_only  — nodes extracted, edges not yet built
      full           — nodes + edges built

    Used by GraphTrigger (Gate 6) to decide whether a graph query is feasible
    without rebuilding. Primary key is project_id (one row per project).
    """

    __tablename__ = "graph_build_status"

    project_id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="not_built")
    node_count = Column(String, default="0")       # stored as string for SQLite compatibility
    edge_count = Column(String, default="0")
    last_built_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "status": self.status,
            "node_count": int(self.node_count or 0),
            "edge_count": int(self.edge_count or 0),
            "last_built_at": self.last_built_at.isoformat() if self.last_built_at else None,
        }
