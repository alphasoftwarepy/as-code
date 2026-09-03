"""
AS Core — Graph Ingestion Pipeline Adapter

Orchestrates deterministic, project-scoped extraction and resolution of
knowledge entities and relationships from document content into GraphStore.

Architecture:
    Document / Content
           ↓
    StructuralExtractor
           ↓
    EntityResolver
           ↓
    GraphStore

Principles:
    - 100% deterministic (no LLM, no embeddings, no external network).
    - Strictly project-scoped (project_id required; no global graph).
    - Fail-safe transaction management (explicit rollback on failure; no partial state).
    - Caller-controlled commit by default (commit=False).
    - Decoupled from RAG ingestion and background tasks.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from runtime.graph.extractor import StructuralExtractor
from runtime.graph.resolver import EntityResolver
from runtime.graph.store import GraphStore

logger = logging.getLogger("as-code.runtime.graph.ingestion")


# ── Ingestion Result Contract ──────────────────────────────────


class IngestionResult(BaseModel):
    """
    Compact result of document graph ingestion.

    Fields:
        status          Outcome status: "success" | "empty" | "failed"
        nodes_created   Number of nodes created or updated with new provenance
        edges_created   Number of edges created or updated
        document_id     ID of the processed document
        project_id      ID of the target project
        error           Error description if status == "failed"
    """

    status: str = Field(description="Outcome: success | empty | failed")
    nodes_created: int = Field(default=0, description="Nodes persisted or merged")
    edges_created: int = Field(default=0, description="Edges persisted or merged")
    document_id: Optional[str] = Field(default=None, description="Document ID")
    project_id: Optional[str] = Field(default=None, description="Project ID")
    error: Optional[str] = Field(default=None, description="Error detail if failed")


# ── Graph Ingestion Pipeline Adapter ──────────────────────────


class GraphIngestionPipeline:
    """
    Adapter orchestrating extraction, entity resolution, and storage
    for the optional Graph capability.
    """

    def __init__(
        self,
        extractor: Optional[StructuralExtractor] = None,
        resolver: Optional[EntityResolver] = None,
        store: Optional[GraphStore] = None,
    ):
        self.store = store or GraphStore()
        self.extractor = extractor or StructuralExtractor()
        self.resolver = resolver or EntityResolver(store=self.store)

    def ingest_document(
        self,
        db: Session,
        project_id: str,
        document_id: str,
        text: str,
        domain: Optional[str] = None,
        commit: bool = False,
        propagate_exceptions: bool = True,
    ) -> IngestionResult:
        """
        Extract and persist graph entities and relationships from document text.

        Args:
            db: Active SQLAlchemy Session injected by caller.
            project_id: Non-empty target project ID.
            document_id: Non-empty document identifier.
            text: Raw textual content of the document.
            domain: Optional domain hint (e.g. 'programming', 'legal').
            commit: If True, calls db.commit() after successful ingestion.
                    Defaults to False (caller controls commit boundary).
            propagate_exceptions: If True, re-raises any caught persistence exception
                                  after performing db.rollback(). Defaults to True.

        Returns:
            IngestionResult summarizing the ingestion outcome.
        """
        # 1. Validate project_id
        if not project_id or not isinstance(project_id, str) or not project_id.strip():
            logger.warning("[GRAPH-INGESTION] Ingestion rejected: invalid or empty project_id")
            return IngestionResult(
                status="failed",
                error="Invalid project_id: must be a non-empty string",
                document_id=document_id if isinstance(document_id, str) else None,
                project_id=None,
            )

        # 2. Validate document_id
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            logger.warning("[GRAPH-INGESTION] Ingestion rejected: invalid or empty document_id")
            return IngestionResult(
                status="failed",
                error="Invalid document_id: must be a non-empty string",
                document_id=None,
                project_id=project_id,
            )

        # 3. Validate and handle empty text
        if not text or not isinstance(text, str) or not text.strip():
            logger.debug(f"[GRAPH-INGESTION] Empty text for doc={document_id} proj={project_id}")
            return IngestionResult(
                status="empty",
                nodes_created=0,
                edges_created=0,
                document_id=document_id,
                project_id=project_id,
            )

        try:
            # 4. Extract entities and relationships
            extraction = self.extractor.extract(
                text=text,
                document_id=document_id,
                domain=domain,
            )

            # If no structural elements were extracted, return controlled empty result
            if not extraction.entities and not extraction.relationships:
                logger.debug(
                    f"[GRAPH-INGESTION] No entities or relationships extracted from doc={document_id}"
                )
                return IngestionResult(
                    status="empty",
                    nodes_created=0,
                    edges_created=0,
                    document_id=document_id,
                    project_id=project_id,
                )

            # 5. Resolve and persist in GraphStore
            nodes_count, edges_count = self.resolver.ingest_extraction_result(
                db=db,
                project_id=project_id,
                extraction=extraction,
            )

            # Optional commit if caller explicitly delegates transaction closure
            if commit:
                db.commit()

            logger.info(
                f"[GRAPH-INGESTION] Completed doc={document_id} proj={project_id}: "
                f"nodes={nodes_count} edges={edges_count}"
            )

            return IngestionResult(
                status="success",
                nodes_created=nodes_count,
                edges_created=edges_count,
                document_id=document_id,
                project_id=project_id,
            )

        except Exception as e:
            # Enforce atomic rollback on failure to prevent partial graph state
            try:
                db.rollback()
                logger.warning(
                    f"[GRAPH-INGESTION] Transaction rolled back for doc={document_id} proj={project_id}"
                )
            except Exception as rb_err:
                logger.error(f"[GRAPH-INGESTION] Rollback execution failed: {rb_err}")

            logger.error(
                f"[GRAPH-INGESTION] Pipeline error for doc={document_id} proj={project_id}: {e}",
                exc_info=True,
            )

            if propagate_exceptions:
                raise

            return IngestionResult(
                status="failed",
                error=str(e),
                document_id=document_id,
                project_id=project_id,
            )

    def ingest_text(
        self,
        db: Session,
        project_id: str,
        document_id: str,
        text: str,
        domain: Optional[str] = None,
        commit: bool = False,
        propagate_exceptions: bool = True,
    ) -> IngestionResult:
        """Convenience alias for ingest_document."""
        return self.ingest_document(
            db=db,
            project_id=project_id,
            document_id=document_id,
            text=text,
            domain=domain,
            commit=commit,
            propagate_exceptions=propagate_exceptions,
        )
