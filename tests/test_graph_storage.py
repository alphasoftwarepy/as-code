"""
AS Core — GATE 3: Graph Storage & Isolation Test Suite

Validates:
  1. CRUD for GraphNode (create, read, find, delete).
  2. CRUD for GraphEdge (create, read, delete).
  3. Strict project_id isolation:
     - Project A never sees Project B nodes or edges.
     - Same entity label in different projects = distinct records.
     - Cross-project edge creation is rejected.
     - Delete project graph affects only the target project.
  4. Idempotency:
     - Saving duplicate node merges provenance and aliases without duplicating records.
     - Saving duplicate edge returns existing edge.
  5. Document provenance:
     - document_id correctly tracked in source_doc_ids and source_doc_id.
  6. Build status lifecycle:
     - not_built -> entities_only -> full.
     - Node and edge counts synced accurately.
  7. Safe empty queries:
     - Querying an empty or non-existent project returns empty results without error.
  8. Contract bridge:
     - to_graph_query_result() correctly builds GraphQueryResult.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.graph_models import GraphBase
from runtime.graph.store import GraphStore


@pytest.fixture
def db_session():
    """In-memory SQLite database isolated per test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    GraphBase.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def store():
    return GraphStore()


# ── CRUD: GraphNode ─────────────────────────────────────────────


class TestNodeCRUD:
    def test_save_and_get_node(self, db_session, store):
        node = store.save_node(
            db=db_session,
            project_id="proj-1",
            label="Juan Pérez",
            entity_type="person",
            source_doc_id="doc-100",
            aliases=["J. Pérez"],
            metadata={"role": "firmante"},
        )
        assert node is not None
        assert node.id is not None
        assert node.label == "Juan Pérez"
        assert node.label_normalized == "juan pérez"
        assert node.entity_type == "person"
        assert node.source_doc_ids == ["doc-100"]
        assert node.aliases == ["J. Pérez"]
        assert node.meta == {"role": "firmante"}

        fetched = store.get_node_by_id(db_session, "proj-1", node.id)
        assert fetched is not None
        assert fetched.id == node.id

    def test_find_node_by_label_case_and_whitespace_insensitive(self, db_session, store):
        store.save_node(
            db=db_session,
            project_id="proj-1",
            label="  Empresa   Alpha S.A.  ",
            entity_type="org",
        )
        found = store.find_node_by_label(db_session, "proj-1", "empresa alpha s.a.")
        assert found is not None
        assert found.label == "  Empresa   Alpha S.A.  "

    def test_delete_node(self, db_session, store):
        node = store.save_node(
            db=db_session, project_id="proj-1", label="Temporal", entity_type="concept"
        )
        assert store.delete_node(db_session, "proj-1", node.id) is True
        assert store.get_node_by_id(db_session, "proj-1", node.id) is None
        assert store.delete_node(db_session, "proj-1", node.id) is False


# ── CRUD: GraphEdge ─────────────────────────────────────────────


class TestEdgeCRUD:
    def test_save_and_list_edges(self, db_session, store):
        n1 = store.save_node(db_session, "proj-1", "A", "module")
        n2 = store.save_node(db_session, "proj-1", "B", "module")

        edge = store.save_edge(
            db=db_session,
            project_id="proj-1",
            source_node_id=n1.id,
            target_node_id=n2.id,
            relation_type="importa",
            source_doc_id="doc-app",
            confidence=0.95,
            metadata={"line": "12"},
        )
        assert edge is not None
        assert edge.relation_type == "importa"
        assert edge.source_doc_id == "doc-app"
        assert edge.confidence == 0.95
        assert edge.meta == {"line": "12"}

        edges = store.list_edges(db_session, "proj-1")
        assert len(edges) == 1
        assert edges[0].id == edge.id

    def test_get_edges_for_node(self, db_session, store):
        n1 = store.save_node(db_session, "proj-1", "Contrato A", "contract")
        n2 = store.save_node(db_session, "proj-1", "Persona X", "person")
        n3 = store.save_node(db_session, "proj-1", "Persona Y", "person")

        store.save_edge(db_session, "proj-1", n2.id, n1.id, "firma")
        store.save_edge(db_session, "proj-1", n3.id, n1.id, "firma")

        n1_edges = store.get_edges_for_node(db_session, "proj-1", n1.id)
        assert len(n1_edges) == 2

    def test_reject_edge_with_missing_or_foreign_node(self, db_session, store):
        n1 = store.save_node(db_session, "proj-1", "Node1", "concept")
        n2 = store.save_node(db_session, "proj-2", "Node2", "concept")

        # Rejected because n2 belongs to proj-2
        edge = store.save_edge(db_session, "proj-1", n1.id, n2.id, "rel")
        assert edge is None

        # Rejected because target node does not exist
        edge_missing = store.save_edge(db_session, "proj-1", n1.id, "non-existent", "rel")
        assert edge_missing is None


# ── Strict Project Isolation ────────────────────────────────────


class TestProjectIsolation:
    def test_nodes_isolated_between_projects(self, db_session, store):
        nA = store.save_node(db_session, "proj-A", "Entity A", "concept")
        nB = store.save_node(db_session, "proj-B", "Entity B", "concept")

        assert store.get_node_by_id(db_session, "proj-A", nB.id) is None
        assert store.get_node_by_id(db_session, "proj-B", nA.id) is None

        listA = store.list_nodes(db_session, "proj-A")
        assert [n.id for n in listA] == [nA.id]

        listB = store.list_nodes(db_session, "proj-B")
        assert [n.id for n in listB] == [nB.id]

    def test_same_label_in_different_projects_are_distinct(self, db_session, store):
        nA = store.save_node(db_session, "proj-A", "Juan Pérez", "person")
        nB = store.save_node(db_session, "proj-B", "Juan Pérez", "person")

        assert nA.id != nB.id
        assert nA.project_id == "proj-A"
        assert nB.project_id == "proj-B"

    def test_delete_project_graph_only_clears_target_project(self, db_session, store):
        nA = store.save_node(db_session, "proj-A", "A", "concept")
        nB = store.save_node(db_session, "proj-B", "B", "concept")

        store.save_edge(db_session, "proj-A", nA.id, nA.id, "self_loop")
        store.save_edge(db_session, "proj-B", nB.id, nB.id, "self_loop")

        store.delete_project_graph(db_session, "proj-A")

        assert store.list_nodes(db_session, "proj-A") == []
        assert store.list_edges(db_session, "proj-A") == []

        # Project B completely unaffected
        assert len(store.list_nodes(db_session, "proj-B")) == 1
        assert len(store.list_edges(db_session, "proj-B")) == 1


# ── Idempotency & Provenance ────────────────────────────────────


class TestIdempotencyAndProvenance:
    def test_node_idempotency_merges_provenance_and_aliases(self, db_session, store):
        n1 = store.save_node(
            db_session, "proj-1", "Contrato Alpha", "contract",
            source_doc_id="doc-1", aliases=["Alpha"]
        )
        n2 = store.save_node(
            db_session, "proj-1", "contrato alpha", "contract",
            source_doc_id="doc-2", aliases=["Alpha V2"]
        )

        assert n1.id == n2.id
        assert set(n2.source_doc_ids) == {"doc-1", "doc-2"}
        assert set(n2.aliases) == {"Alpha", "Alpha V2"}

        nodes = store.list_nodes(db_session, "proj-1")
        assert len(nodes) == 1

    def test_edge_idempotency_does_not_duplicate(self, db_session, store):
        n1 = store.save_node(db_session, "proj-1", "X", "concept")
        n2 = store.save_node(db_session, "proj-1", "Y", "concept")

        e1 = store.save_edge(db_session, "proj-1", n1.id, n2.id, "connects", source_doc_id="doc-1")
        e2 = store.save_edge(db_session, "proj-1", n1.id, n2.id, "connects", source_doc_id="doc-1")

        assert e1.id == e2.id
        edges = store.list_edges(db_session, "proj-1")
        assert len(edges) == 1


# ── Build Status & Safe Defaults ────────────────────────────────


class TestBuildStatusAndBridge:
    def test_default_build_status_is_not_built(self, db_session, store):
        status = store.get_build_status(db_session, "empty-proj")
        assert status.status == "not_built"
        assert status.node_count == 0 or status.node_count == "0"

    def test_update_build_status_syncs_counts(self, db_session, store):
        n1 = store.save_node(db_session, "proj-1", "N1", "concept")
        n2 = store.save_node(db_session, "proj-1", "N2", "concept")
        store.save_edge(db_session, "proj-1", n1.id, n2.id, "link")

        store.update_build_status(db_session, "proj-1", "full")

        status = store.get_build_status(db_session, "proj-1")
        assert status.status == "full"
        assert int(status.node_count) == 2
        assert int(status.edge_count) == 1
        assert status.last_built_at is not None

    def test_empty_project_query_result(self, db_session, store):
        result = store.to_graph_query_result(db_session, "empty-proj")
        assert result.graph_available is True
        assert result.entities == []
        assert result.relationships == []
        assert result.source_document_ids == []

    def test_to_graph_query_result_converts_correctly(self, db_session, store):
        n1 = store.save_node(db_session, "proj-1", "A", "person", source_doc_id="doc-1")
        n2 = store.save_node(db_session, "proj-1", "B", "org", source_doc_id="doc-2")
        store.save_edge(db_session, "proj-1", n1.id, n2.id, "trabaja_en", source_doc_id="doc-1")

        res = store.to_graph_query_result(db_session, "proj-1")
        assert res.graph_available is True
        assert len(res.entities) == 2
        assert len(res.relationships) == 1
        assert set(res.source_document_ids) == {"doc-1", "doc-2"}
        assert res.relationships[0].relation_type == "trabaja_en"
