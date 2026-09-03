"""
AS Code — Graph Query Engine & Bounded Traversal

Implements deterministic, bounded relational query and multi-hop graph traversal.
Operates exclusively through GraphStore and returns a self-contained GraphQueryResult.
Zero dependencies on NetworkX or external graph libraries.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from api.graph_models import GraphEdge, GraphNode
from runtime.graph.contracts import (
    GraphEntity,
    GraphProvider,
    GraphQuery,
    GraphQueryResult,
    GraphRelationship,
)
from runtime.graph.normalizer import normalize_key, normalize_label
from runtime.graph.store import GraphStore

logger = logging.getLogger("as-code.runtime.graph.query")


class GraphQueryEngine(GraphProvider):
    """
    Deterministic Query Engine with Bounded BFS Traversal.
    Implements the GraphProvider abstract interface.
    """

    def __init__(self, store: Optional[GraphStore] = None):
        self.store = store or GraphStore()

    def is_available(self) -> bool:
        """Lightweight check: verify store is instantiated."""
        return self.store is not None

    def query(self, graph_query: GraphQuery, db: Optional[Session] = None) -> GraphQueryResult:
        """
        Execute a bounded relational query over the project's graph.
        Guarantees:
          - Strict project_id isolation.
          - Deterministic starting point resolution (no random tie-breaking).
          - Strict bounds: max_depth, max_nodes, timeout_seconds.
          - Loop / cycle protection via visited set.
          - Never raises exceptions (degrades safely).
        """
        start_time = time.perf_counter()

        if not db:
            logger.warning("[GRAPH-QUERY] No database session provided to query engine.")
            return GraphQueryResult.unavailable()

        project_id = graph_query.project_id
        if not project_id:
            logger.warning("[GRAPH-QUERY] Missing project_id in GraphQuery.")
            return GraphQueryResult.unavailable()

        try:
            # 1. Verify build status for project
            build_status = self.store.get_build_status(db, project_id)
            if build_status.status == "not_built":
                logger.debug(f"[GRAPH-QUERY] Graph not built for project_id={project_id}")
                return GraphQueryResult.empty()

            # 2. Resolve starting seed nodes deterministically
            seeds = self._resolve_seed_nodes(db, project_id, graph_query.query, graph_query.domain)
            if not seeds:
                logger.debug(f"[GRAPH-QUERY] No seed nodes resolved for query={graph_query.query!r}")
                return GraphQueryResult.empty()

            # 3. Bounded BFS traversal
            entities, relationships, source_docs = self._traverse_bounded(
                db=db,
                project_id=project_id,
                seeds=seeds,
                max_depth=graph_query.max_depth,
                max_nodes=graph_query.max_nodes,
                timeout_seconds=graph_query.timeout_seconds,
                start_time=start_time,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"[GRAPH-QUERY] Traversal complete for project_id={project_id}: "
                f"nodes={len(entities)} edges={len(relationships)} in {elapsed_ms:.2f}ms"
            )

            return GraphQueryResult(
                entities=entities,
                relationships=relationships,
                source_document_ids=source_docs,
                graph_available=True,
                metadata={
                    "seeds_count": str(len(seeds)),
                    "traversal_depth": str(graph_query.max_depth),
                    "elapsed_ms": f"{elapsed_ms:.2f}",
                },
            )

        except Exception as e:
            logger.error(f"[GRAPH-QUERY] Query failed unexpectedly: {e}", exc_info=True)
            return GraphQueryResult.unavailable()

    def _resolve_seed_nodes(
        self,
        db: Session,
        project_id: str,
        query_text: str,
        domain: Optional[str] = None,
    ) -> List[GraphNode]:
        """
        Deterministic starting point resolution.
        Priority:
          1. Direct exact match of normalized query to node label.
          2. Exact match of normalized query to alias.
          3. Substring containment of node label in query text (sorted by longest label first).
        If multiple ambiguous candidates match without distinction, conservative behavior applies.
        """
        if not query_text or not query_text.strip():
            return []

        all_nodes = self.store.list_nodes(db, project_id)
        if not all_nodes:
            return []

        # Sort all nodes deterministically by label_normalized, then id
        all_nodes = sorted(all_nodes, key=lambda n: (n.label_normalized, n.id))

        query_norm = normalize_label(query_text)
        query_key = normalize_key(query_text)

        # 1. Exact match on label
        exact_matches = [n for n in all_nodes if n.label_normalized == query_norm]
        if exact_matches:
            return self._filter_by_domain(exact_matches, domain)

        # 2. Exact match on aliases
        alias_matches = [
            n for n in all_nodes
            if any(normalize_label(a) == query_norm for a in n.aliases)
        ]
        if alias_matches:
            return self._filter_by_domain(alias_matches, domain)

        # 3. Stripped key match
        key_matches = [n for n in all_nodes if normalize_key(n.label) == query_key and query_key]
        if key_matches:
            return self._filter_by_domain(key_matches, domain)

        # 4. Containment: find nodes whose label or aliases appear as complete phrases in query_text
        containment_matches = []
        for n in all_nodes:
            lbl_norm = n.label_normalized
            # Avoid single character false positives
            if len(lbl_norm) >= 3 and lbl_norm in query_norm:
                containment_matches.append(n)
            else:
                for a in n.aliases:
                    a_norm = normalize_label(a)
                    if len(a_norm) >= 3 and a_norm in query_norm:
                        containment_matches.append(n)
                        break

        if containment_matches:
            # Sort deterministically: longest label match first, then alphabetical
            containment_matches.sort(key=lambda n: (-len(n.label_normalized), n.label_normalized, n.id))
            return self._filter_by_domain(containment_matches, domain)

        return []

    def _filter_by_domain(self, nodes: List[GraphNode], domain: Optional[str]) -> List[GraphNode]:
        """Prioritize nodes matching domain if domain hint is provided, otherwise return nodes."""
        if not domain or not nodes:
            return nodes

        domain_lower = domain.lower()
        domain_matched = [n for n in nodes if n.entity_type == domain_lower or domain_lower in str(n.meta)]
        return domain_matched if domain_matched else nodes

    def _traverse_bounded(
        self,
        db: Session,
        project_id: str,
        seeds: List[GraphNode],
        max_depth: int,
        max_nodes: int,
        timeout_seconds: float,
        start_time: float,
    ) -> Tuple[List[GraphEntity], List[GraphRelationship], List[str]]:
        """
        Bounded Breadth-First Search (BFS) traversal.
        - Limits maximum depth (hops).
        - Limits total nodes visited (max_nodes).
        - Prevents cycles using a visited set.
        - Respects timeout_seconds.
        - Enforces deterministic ordering.
        """
        visited_node_ids: Set[str] = set()
        visited_nodes: Dict[str, GraphNode] = {}
        collected_edges: Dict[str, GraphEdge] = {}
        source_doc_ids: Set[str] = set()

        # Queue contains tuples of (node, current_depth)
        queue: deque[Tuple[GraphNode, int]] = deque()

        for seed in seeds:
            if len(visited_node_ids) >= max_nodes:
                break
            if seed.id not in visited_node_ids:
                visited_node_ids.add(seed.id)
                visited_nodes[seed.id] = seed
                queue.append((seed, 0))
                source_doc_ids.update(seed.source_doc_ids)

        while queue:
            # Check timeout
            if (time.perf_counter() - start_time) > timeout_seconds:
                logger.warning(f"[GRAPH-QUERY] Traversal reached timeout ({timeout_seconds}s). Returning partial results.")
                break

            current_node, current_depth = queue.popleft()

            # If depth limit reached for this branch, don't expand neighbors
            if current_depth >= max_depth:
                continue

            # Query edges connected to current_node in this project
            edges = self.store.get_edges_for_node(db, project_id, current_node.id)

            # Sort edges deterministically by relation_type, then id
            edges = sorted(edges, key=lambda e: (e.relation_type, e.id))

            for edge in edges:
                # Check timeout during edge loop
                if (time.perf_counter() - start_time) > timeout_seconds:
                    break

                collected_edges[edge.id] = edge
                if edge.source_doc_id:
                    source_doc_ids.add(edge.source_doc_id)

                # Determine neighbor id
                neighbor_id = edge.target_node_id if edge.source_node_id == current_node.id else edge.source_node_id

                # Cycle check / node budget check
                if neighbor_id not in visited_node_ids:
                    if len(visited_node_ids) >= max_nodes:
                        logger.debug(f"[GRAPH-QUERY] Reached max_nodes={max_nodes}. Halting node expansion.")
                        break

                    neighbor = self.store.get_node_by_id(db, project_id, neighbor_id)
                    if neighbor:
                        visited_node_ids.add(neighbor.id)
                        visited_nodes[neighbor.id] = neighbor
                        source_doc_ids.update(neighbor.source_doc_ids)
                        queue.append((neighbor, current_depth + 1))

        # Convert to contract types with deterministic sorting
        entities = [
            GraphEntity(
                id=n.id,
                label=n.label,
                entity_type=n.entity_type,
                source_document_ids=sorted(n.source_doc_ids),
                aliases=sorted(n.aliases),
                metadata={str(k): str(v) for k, v in n.meta.items()},
            )
            for n in sorted(visited_nodes.values(), key=lambda n: (n.label_normalized, n.id))
        ]

        relationships = [
            GraphRelationship(
                source_id=e.source_node_id,
                target_id=e.target_node_id,
                relation_type=e.relation_type,
                source_document_id=e.source_doc_id,
                confidence=e.confidence,
                metadata={str(k): str(v) for k, v in e.meta.items()},
            )
            for e in sorted(collected_edges.values(), key=lambda e: (e.relation_type, e.id))
        ]

        return entities, relationships, sorted(source_doc_ids)
