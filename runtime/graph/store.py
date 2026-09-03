"""
AS Code — Graph Store

Encapsulates all SQLite persistence for the optional Graph layer.

IMPORTANT: This is the ONLY module that should access graph_nodes, graph_edges,
and graph_build_status tables directly. All other code must go through GraphStore.

Design rules:
  - Every method filters by project_id. No cross-project access is possible.
  - session_id is NOT accepted as a parameter — Graph ownership is per-project.
  - Idempotency: save_node() and save_edge() use INSERT-OR-IGNORE semantics via
    the UniqueConstraints defined in graph_models.py. Duplicate calls on the
    same structural key return the existing record instead of raising.
  - Failures are caught and logged; callers receive None / [] instead of exceptions.
  - No NetworkX, no FAISS, no RAG imports.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from api.graph_models import GraphBuildStatus, GraphEdge, GraphNode
from runtime.graph.contracts import (
    GraphEntity,
    GraphQueryResult,
    GraphRelationship,
)

logger = logging.getLogger("as-code.runtime.graph.store")


# ── Normalization helper ────────────────────────────────────────


def _normalize_label(label: str) -> str:
    """
    Produce a canonical form of an entity label for idempotency checks.

    Rules:
      - Unicode NFC normalization
      - Lowercase
      - Strip leading/trailing whitespace
      - Collapse internal whitespace to single space

    This is intentionally minimal — full entity resolution (fuzzy matching,
    alias merging, LLM disambiguation) belongs to Gate 4 (EntityResolver).
    """
    normalized = unicodedata.normalize("NFC", label)
    return " ".join(normalized.lower().strip().split())


# ── GraphStore ──────────────────────────────────────────────────


class GraphStore:
    """
    Thin persistence layer for Graph nodes and edges.

    All operations accept a SQLAlchemy Session injected by the caller.
    GraphStore itself holds no state — it is stateless and reusable.
    """

    # ── Nodes ───────────────────────────────────────────────────

    def save_node(
        self,
        db: Session,
        project_id: str,
        label: str,
        entity_type: str,
        source_doc_id: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[GraphNode]:
        """
        Insert a new node or return the existing one (idempotent).

        Idempotency key: (project_id, label_normalized).
        If the node already exists:
          - Updates source_doc_ids to include the new document (provenance merge).
          - Merges aliases without duplication.
        """
        try:
            label_normalized = _normalize_label(label)

            existing = (
                db.query(GraphNode)
                .filter_by(project_id=project_id, label_normalized=label_normalized)
                .first()
            )

            if existing:
                # Provenance merge — add new doc_id if not already present
                if source_doc_id:
                    current_docs = existing.source_doc_ids
                    if source_doc_id not in current_docs:
                        current_docs.append(source_doc_id)
                        existing.source_doc_ids = current_docs

                # Alias merge
                if aliases:
                    current_aliases = existing.aliases
                    merged = list(set(current_aliases + aliases))
                    existing.aliases = merged

                db.flush()
                return existing

            # New node
            node = GraphNode(
                project_id=project_id,
                label=label,
                label_normalized=label_normalized,
                entity_type=entity_type,
            )
            node.source_doc_ids = [source_doc_id] if source_doc_id else []
            node.aliases = aliases or []
            node.meta = metadata or {}

            db.add(node)
            db.flush()
            logger.debug(
                f"[GRAPH-STORE] Node created: project={project_id} label={label!r} type={entity_type}"
            )
            return node

        except Exception as e:
            logger.error(f"[GRAPH-STORE] save_node failed: {e}", exc_info=True)
            return None

    def get_node_by_id(self, db: Session, project_id: str, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID, enforcing project_id isolation."""
        try:
            return db.query(GraphNode).filter_by(id=node_id, project_id=project_id).first()
        except Exception as e:
            logger.error(f"[GRAPH-STORE] get_node_by_id failed: {e}", exc_info=True)
            return None

    def find_node_by_label(
        self, db: Session, project_id: str, label: str
    ) -> Optional[GraphNode]:
        """Find a node by exact normalized label within a project."""
        try:
            label_normalized = _normalize_label(label)
            return (
                db.query(GraphNode)
                .filter_by(project_id=project_id, label_normalized=label_normalized)
                .first()
            )
        except Exception as e:
            logger.error(f"[GRAPH-STORE] find_node_by_label failed: {e}", exc_info=True)
            return None

    def list_nodes(self, db: Session, project_id: str) -> List[GraphNode]:
        """Return all nodes belonging to a project."""
        try:
            return db.query(GraphNode).filter_by(project_id=project_id).all()
        except Exception as e:
            logger.error(f"[GRAPH-STORE] list_nodes failed: {e}", exc_info=True)
            return []

    def delete_node(self, db: Session, project_id: str, node_id: str) -> bool:
        """Delete a single node by ID, enforcing project isolation."""
        try:
            node = db.query(GraphNode).filter_by(id=node_id, project_id=project_id).first()
            if node:
                db.delete(node)
                db.flush()
                return True
            return False
        except Exception as e:
            logger.error(f"[GRAPH-STORE] delete_node failed: {e}", exc_info=True)
            return False

    # ── Edges ───────────────────────────────────────────────────

    def save_edge(
        self,
        db: Session,
        project_id: str,
        source_node_id: str,
        target_node_id: str,
        relation_type: str,
        source_doc_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> Optional[GraphEdge]:
        """
        Insert an edge or return the existing one (idempotent).

        Idempotency key: (project_id, source_node_id, target_node_id, relation_type, source_doc_id).
        Both nodes must belong to project_id — this is a structural invariant.
        """
        try:
            # Verify both nodes belong to this project
            src = db.query(GraphNode).filter_by(id=source_node_id, project_id=project_id).first()
            tgt = db.query(GraphNode).filter_by(id=target_node_id, project_id=project_id).first()
            if not src or not tgt:
                logger.warning(
                    f"[GRAPH-STORE] save_edge rejected: node(s) not found in project={project_id} "
                    f"src={source_node_id} tgt={target_node_id}"
                )
                return None

            existing = (
                db.query(GraphEdge)
                .filter_by(
                    project_id=project_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    relation_type=relation_type,
                    source_doc_id=source_doc_id,
                )
                .first()
            )
            if existing:
                return existing

            edge = GraphEdge(
                project_id=project_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relation_type=relation_type,
                source_doc_id=source_doc_id,
                confidence=confidence,
            )
            edge.meta = metadata or {}
            db.add(edge)
            db.flush()
            logger.debug(
                f"[GRAPH-STORE] Edge created: project={project_id} "
                f"{source_node_id} --[{relation_type}]--> {target_node_id}"
            )
            return edge

        except Exception as e:
            logger.error(f"[GRAPH-STORE] save_edge failed: {e}", exc_info=True)
            return None

    def get_edges_for_node(
        self, db: Session, project_id: str, node_id: str
    ) -> List[GraphEdge]:
        """Return all edges where the given node is source or target, scoped to project."""
        try:
            return (
                db.query(GraphEdge)
                .filter(
                    GraphEdge.project_id == project_id,
                    (GraphEdge.source_node_id == node_id) | (GraphEdge.target_node_id == node_id),
                )
                .all()
            )
        except Exception as e:
            logger.error(f"[GRAPH-STORE] get_edges_for_node failed: {e}", exc_info=True)
            return []

    def list_edges(self, db: Session, project_id: str) -> List[GraphEdge]:
        """Return all edges belonging to a project."""
        try:
            return db.query(GraphEdge).filter_by(project_id=project_id).all()
        except Exception as e:
            logger.error(f"[GRAPH-STORE] list_edges failed: {e}", exc_info=True)
            return []

    def delete_edge(self, db: Session, project_id: str, edge_id: str) -> bool:
        """Delete a single edge by ID, enforcing project isolation."""
        try:
            edge = db.query(GraphEdge).filter_by(id=edge_id, project_id=project_id).first()
            if edge:
                db.delete(edge)
                db.flush()
                return True
            return False
        except Exception as e:
            logger.error(f"[GRAPH-STORE] delete_edge failed: {e}", exc_info=True)
            return False

    # ── Build Status ────────────────────────────────────────────

    def get_build_status(self, db: Session, project_id: str) -> GraphBuildStatus:
        """
        Return the build status for a project. Never raises — returns a
        synthetic 'not_built' status if no record exists.
        """
        try:
            status = db.query(GraphBuildStatus).filter_by(project_id=project_id).first()
            if status is None:
                return GraphBuildStatus(project_id=project_id, status="not_built", node_count="0", edge_count="0")
            return status
        except Exception as e:
            logger.error(f"[GRAPH-STORE] get_build_status failed: {e}", exc_info=True)
            return GraphBuildStatus(project_id=project_id, status="not_built", node_count="0", edge_count="0")

    def update_build_status(
        self,
        db: Session,
        project_id: str,
        status: str,
    ) -> None:
        """
        Upsert the build status for a project and sync node/edge counts.

        status must be one of: 'not_built' | 'entities_only' | 'full'
        """
        try:
            node_count = db.query(GraphNode).filter_by(project_id=project_id).count()
            edge_count = db.query(GraphEdge).filter_by(project_id=project_id).count()

            record = db.query(GraphBuildStatus).filter_by(project_id=project_id).first()
            if record:
                record.status = status
                record.node_count = str(node_count)
                record.edge_count = str(edge_count)
                record.last_built_at = datetime.utcnow()
            else:
                record = GraphBuildStatus(
                    project_id=project_id,
                    status=status,
                    node_count=str(node_count),
                    edge_count=str(edge_count),
                    last_built_at=datetime.utcnow(),
                )
                db.add(record)
            db.flush()
        except Exception as e:
            logger.error(f"[GRAPH-STORE] update_build_status failed: {e}", exc_info=True)

    # ── Project-level operations ─────────────────────────────────

    def delete_project_graph(self, db: Session, project_id: str) -> None:
        """
        Delete ALL graph data for a project (nodes, edges, build status).
        Only affects the specified project_id.
        """
        try:
            db.query(GraphEdge).filter_by(project_id=project_id).delete()
            db.query(GraphNode).filter_by(project_id=project_id).delete()
            db.query(GraphBuildStatus).filter_by(project_id=project_id).delete()
            db.flush()
            logger.info(f"[GRAPH-STORE] Deleted all graph data for project={project_id}")
        except Exception as e:
            logger.error(f"[GRAPH-STORE] delete_project_graph failed: {e}", exc_info=True)

    # ── Contract bridge ─────────────────────────────────────────

    def to_graph_query_result(
        self, db: Session, project_id: str
    ) -> GraphQueryResult:
        """
        Convert all stored nodes and edges for a project into a GraphQueryResult.

        This is a simple full-project dump — the Gate 5 query engine will
        implement filtered, bounded traversal. This method supports Gate 3 tests.
        """
        try:
            nodes = self.list_nodes(db, project_id)
            edges = self.list_edges(db, project_id)

            entities = [
                GraphEntity(
                    id=n.id,
                    label=n.label,
                    entity_type=n.entity_type,
                    source_document_ids=n.source_doc_ids,
                    aliases=n.aliases,
                    metadata=n.meta,
                )
                for n in nodes
            ]

            relationships = [
                GraphRelationship(
                    source_id=e.source_node_id,
                    target_id=e.target_node_id,
                    relation_type=e.relation_type,
                    source_document_id=e.source_doc_id,
                    confidence=e.confidence,
                    metadata=e.meta,
                )
                for e in edges
            ]

            source_doc_ids = list(
                {doc_id for n in nodes for doc_id in n.source_doc_ids}
            )

            return GraphQueryResult(
                entities=entities,
                relationships=relationships,
                source_document_ids=source_doc_ids,
                graph_available=True,
            )

        except Exception as e:
            logger.error(f"[GRAPH-STORE] to_graph_query_result failed: {e}", exc_info=True)
            return GraphQueryResult.unavailable()
