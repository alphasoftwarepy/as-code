"""
AS Core — Graph Contracts

Abstract data contracts for the optional Graph capability.

IMPORTANT: This module is intentionally decoupled from every AS Core subsystem.
It does NOT import from:
  - RAG (FAISS, BM25, embeddings, rag_service)
  - Working Memory
  - Coordinator internals
  - SQLAlchemy / SQLite
  - NetworkX
  - Skill loaders

These contracts define the *boundary* of the Graph layer:
  what goes in (GraphQuery), what comes out (GraphQueryResult),
  and the abstract provider interface (GraphProvider).

The concrete implementation lives in later gates (storage, extraction, query).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Entity ─────────────────────────────────────────────────────


class GraphEntity(BaseModel):
    """
    A named entity extracted from one or more project documents.

    Fields:
        id            Stable identifier for this entity (UUID).
        label         Canonical, normalized display name (e.g. "Juan Pérez").
        entity_type   Generic semantic type: person | org | place | date |
                      concept | file | module | contract | product | other.
                      Domain-specific subtypes are carried via metadata.
        source_document_ids
                      IDs of the documents where this entity was found.
                      Drives provenance and isolation checks.
        aliases       Alternative surface forms found in documents
                      (e.g. ["J. Pérez", "Juan A. Pérez"]).
                      Used by entity resolution to avoid duplicates.
        metadata      Arbitrary key-value bag for domain hints or extras.
                      Never contains full document text.
    """

    id: str
    label: str
    entity_type: str
    source_document_ids: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)


# ── Relationship ────────────────────────────────────────────────


class GraphRelationship(BaseModel):
    """
    A directed relationship between two entities found in project documents.

    Fields:
        source_id       ID of the source GraphEntity.
        target_id       ID of the target GraphEntity.
        relation_type   Generic label for the relationship (e.g. "firma",
                        "importa", "propietario", "aparece_con").
                        Domain emerges from the value; no subclasses needed.
        source_document_id
                        ID of the document where this relationship was found.
                        Optional: some relationships may be inferred across docs.
        confidence      Extraction confidence in [0.0, 1.0].
                        1.0 for deterministic extractions; lower for fuzzy.
        metadata        Arbitrary extras (e.g. context snippet reference).
    """

    source_id: str
    target_id: str
    relation_type: str
    source_document_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, str] = Field(default_factory=dict)


# ── Query ───────────────────────────────────────────────────────


class GraphQuery(BaseModel):
    """
    A request for relational reasoning over a project's knowledge graph.

    Fields:
        project_id    MANDATORY. Scopes the query to a single project.
                      The provider must NEVER return knowledge from
                      another project. This is an architectural invariant.
        query         The user's original query text (used for relevance
                      filtering and traversal seed selection).
        domain        Optional hint from the active Skill (e.g. "legal",
                      "programming"). Allows the provider to prioritize
                      relevant entity types. Graph stays generic; this is
                      only a hint.
        max_depth     Maximum traversal hops from seed entities.
                      Default: 2.  Hard cap enforced by the provider.
        max_nodes     Maximum number of entities to return.
                      Default: 30. Prevents memory/latency spikes.
        timeout_seconds
                      Maximum seconds the provider may spend on this query.
                      If exceeded, it must return whatever it has gathered
                      so far (partial result), never raise an exception.
                      Default: 5.0.

    Fields intentionally NOT included:
        session_id    Session is NOT the ownership scope for Graph. Chats
                      within the same project share the same graph.
        relevant_entities
                      Unknown at query-formulation time. The provider
                      extracts seeds from the query text internally.
    """

    project_id: str
    query: str
    domain: Optional[str] = None
    max_depth: int = 2
    max_nodes: int = 30
    timeout_seconds: float = 5.0


# ── Result ──────────────────────────────────────────────────────


class GraphQueryResult(BaseModel):
    """
    Structured output from a GraphProvider.query() call.

    This object is serializable to plain JSON and contains no internal
    runtime state (no NetworkX objects, no SQLAlchemy rows, no FAISS vectors).

    Fields:
        entities            Entities found relevant to the query.
        relationships       Relationships between returned entities.
        source_document_ids Distinct document IDs that contributed to
                            this result. Used for provenance display.
        graph_available     False when the graph system was not available,
                            not built yet, or encountered a non-recoverable
                            error. When False, entities and relationships
                            will be empty; the caller degrades to RAG-only.
        metadata            Optional extras for observability
                            (e.g. traversal_depth reached, node_count_scanned).

    Fields intentionally NOT included:
        paths               Traversal path details are implementation-specific.
                            The formatter decides how to present relationships.
        confidence          A single result-level confidence is misleading;
                            per-relationship confidence is carried in
                            GraphRelationship.confidence.
    """

    entities: List[GraphEntity] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
    source_document_ids: List[str] = Field(default_factory=list)
    graph_available: bool = True
    metadata: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def unavailable(cls) -> "GraphQueryResult":
        """
        Factory for the canonical 'graph not available' result.
        Used by fallback paths so callers never need to construct this manually.
        """
        return cls(graph_available=False)

    @classmethod
    def empty(cls) -> "GraphQueryResult":
        """
        Factory for a valid but empty result (graph built, no matches found).
        """
        return cls(graph_available=True)


# ── Provider (Abstract) ─────────────────────────────────────────


class GraphProvider(ABC):
    """
    Abstract interface for all Graph implementations.

    The coordinator and context assembler only interact with this interface.
    The concrete implementation (SQLite + NetworkX, or any future backend)
    is hidden behind it, satisfying the Dependency Inversion Principle.

    Availability contract:
        - is_available() MUST NOT raise exceptions.
        - query() MUST NOT raise exceptions; on any internal failure it
          returns GraphQueryResult.unavailable() so the caller can
          degrade gracefully to RAG-only context.
        - query() MUST respect GraphQuery.timeout_seconds.
        - query() MUST filter strictly by GraphQuery.project_id;
          results from other projects are an architectural violation.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if this provider is ready to serve queries.

        Lightweight check only (e.g. verify the store is initialized).
        Must not perform I/O that could block for more than a few ms.
        """

    @abstractmethod
    def query(self, graph_query: GraphQuery) -> GraphQueryResult:
        """
        Execute a relational query and return structured results.

        Guaranteed contract:
          - Never raises an exception (all errors → GraphQueryResult.unavailable()).
          - Respects graph_query.timeout_seconds.
          - Respects graph_query.project_id isolation.
          - Returns GraphQueryResult.empty() if graph is built but no
            entities match the query.
        """
