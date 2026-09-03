"""
AS Code — GATE 5: Graph Query Engine & Bounded Traversal Test Suite

Validates:
  1. Basic Queries:
     - Query with existing entity returns seed node.
     - Non-existent entity returns safe empty result (graph_available=True).
     - Empty project / not built returns safe empty result.
  2. Traversal Depth (max_depth):
     - depth 0: only seed nodes, no neighbors.
     - depth 1: seed + immediate 1-hop neighbors.
     - depth 2: 2-hop multi-branch neighbors.
  3. Bounded Cognition & Limits:
     - Cycle termination (A -> B -> C -> A does not infinite loop).
     - max_nodes cap strictly respected.
     - timeout_seconds partial results protection.
  4. Project Isolation:
     - Project A query NEVER returns Project B nodes or relationships.
     - Query on identical entity label in Project A returns only Project A subgraph.
  5. Determinism:
     - Repeated queries return identical results (order and content).
  6. Provenance & Explainability:
     - source_document_ids correctly aggregated from visited nodes and edges.
"""

import time
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.graph_models import GraphBase
from runtime.graph.contracts import GraphQuery
from runtime.graph.query import GraphQueryEngine
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


@pytest.fixture
def engine(store):
    return GraphQueryEngine(store=store)


# ── 1. Basic Query Tests ─────────────────────────────────────────


class TestBasicQueries:
    def test_query_existing_entity(self, db_session, engine, store):
        n1 = store.save_node(db_session, "proj-1", "Carlos Pérez", "person", source_doc_id="doc-1")
        store.update_build_status(db_session, "proj-1", "entities_only")

        q = GraphQuery(project_id="proj-1", query="Carlos Pérez", max_depth=0)
        res = engine.query(q, db=db_session)

        assert res.graph_available is True
        assert len(res.entities) == 1
        assert res.entities[0].id == n1.id
        assert res.entities[0].label == "Carlos Pérez"
        assert res.source_document_ids == ["doc-1"]

    def test_query_non_existent_entity_returns_empty_safely(self, db_session, engine, store):
        store.save_node(db_session, "proj-1", "Alpha", "org")
        store.update_build_status(db_session, "proj-1", "entities_only")

        q = GraphQuery(project_id="proj-1", query="Beta Inexistente")
        res = engine.query(q, db=db_session)

        assert res.graph_available is True
        assert res.entities == []
        assert res.relationships == []
        assert res.source_document_ids == []

    def test_query_unbuilt_or_empty_project(self, db_session, engine):
        q = GraphQuery(project_id="unbuilt-proj", query="Cualquier Cosa")
        res = engine.query(q, db=db_session)

        assert res.graph_available is True
        assert res.entities == []
        assert res.relationships == []

    def test_query_without_db_returns_unavailable(self, engine):
        q = GraphQuery(project_id="p1", query="test")
        res = engine.query(q, db=None)
        assert res.graph_available is False


# ── 2. Traversal Depth Tests ────────────────────────────────────


class TestTraversalDepth:
    @pytest.fixture
    def linear_graph(self, db_session, store):
        # A --[r1]--> B --[r2]--> C --[r3]--> D
        nA = store.save_node(db_session, "proj-net", "A", "concept", source_doc_id="doc-a")
        nB = store.save_node(db_session, "proj-net", "B", "concept", source_doc_id="doc-b")
        nC = store.save_node(db_session, "proj-net", "C", "concept", source_doc_id="doc-c")
        nD = store.save_node(db_session, "proj-net", "D", "concept", source_doc_id="doc-d")

        store.save_edge(db_session, "proj-net", nA.id, nB.id, "r1", source_doc_id="doc-ab")
        store.save_edge(db_session, "proj-net", nB.id, nC.id, "r2", source_doc_id="doc-bc")
        store.save_edge(db_session, "proj-net", nC.id, nD.id, "r3", source_doc_id="doc-cd")
        store.update_build_status(db_session, "proj-net", "full")

        return {"A": nA, "B": nB, "C": nC, "D": nD}

    def test_depth_zero(self, db_session, engine, linear_graph):
        q = GraphQuery(project_id="proj-net", query="A", max_depth=0)
        res = engine.query(q, db=db_session)

        assert len(res.entities) == 1
        assert res.entities[0].label == "A"
        assert res.relationships == []
        assert res.source_document_ids == ["doc-a"]

    def test_depth_one(self, db_session, engine, linear_graph):
        q = GraphQuery(project_id="proj-net", query="A", max_depth=1)
        res = engine.query(q, db=db_session)

        labels = {e.label for e in res.entities}
        assert labels == {"A", "B"}
        assert len(res.relationships) == 1
        assert res.relationships[0].relation_type == "r1"
        assert set(res.source_document_ids) == {"doc-a", "doc-b", "doc-ab"}

    def test_depth_two(self, db_session, engine, linear_graph):
        q = GraphQuery(project_id="proj-net", query="A", max_depth=2)
        res = engine.query(q, db=db_session)

        labels = {e.label for e in res.entities}
        assert labels == {"A", "B", "C"}
        assert len(res.relationships) == 2
        rel_types = {r.relation_type for r in res.relationships}
        assert rel_types == {"r1", "r2"}


# ── 3. Cycle & Bounded Limits Tests ─────────────────────────────


class TestBoundedCognition:
    def test_cycle_termination(self, db_session, engine, store):
        # Cycle: A -> B -> C -> A
        nA = store.save_node(db_session, "proj-cyc", "NodeA", "concept")
        nB = store.save_node(db_session, "proj-cyc", "NodeB", "concept")
        nC = store.save_node(db_session, "proj-cyc", "NodeC", "concept")

        store.save_edge(db_session, "proj-cyc", nA.id, nB.id, "links")
        store.save_edge(db_session, "proj-cyc", nB.id, nC.id, "links")
        store.save_edge(db_session, "proj-cyc", nC.id, nA.id, "links")
        store.update_build_status(db_session, "proj-cyc", "full")

        # Deep traversal on cycle
        q = GraphQuery(project_id="proj-cyc", query="NodeA", max_depth=10)
        res = engine.query(q, db=db_session)

        # Must terminate and contain exactly 3 nodes
        assert len(res.entities) == 3
        labels = {e.label for e in res.entities}
        assert labels == {"NodeA", "NodeB", "NodeC"}
        assert len(res.relationships) == 3

    def test_max_nodes_limit_respected(self, db_session, engine, store):
        # Hub node connected to 10 satellite nodes
        hub = store.save_node(db_session, "proj-hub", "Hub", "org")
        for i in range(10):
            sat = store.save_node(db_session, "proj-hub", f"Sat_{i}", "person")
            store.save_edge(db_session, "proj-hub", hub.id, sat.id, "connected_to")
        store.update_build_status(db_session, "proj-hub", "full")

        # Bound max_nodes to 4
        q = GraphQuery(project_id="proj-hub", query="Hub", max_depth=2, max_nodes=4)
        res = engine.query(q, db=db_session)

        assert len(res.entities) == 4

    def test_timeout_seconds_safety(self, db_session, engine, store):
        store.save_node(db_session, "proj-t", "Quick", "concept")
        store.update_build_status(db_session, "proj-t", "entities_only")

        # Normal query completes well under timeout
        q = GraphQuery(project_id="proj-t", query="Quick", timeout_seconds=1.0)
        res = engine.query(q, db=db_session)
        assert res.graph_available is True
        assert len(res.entities) == 1


# ── 4. Strict Project Isolation Tests ───────────────────────────


class TestProjectIsolationInQuery:
    def test_isolation_between_projects_with_same_entity_labels(self, db_session, engine, store):
        # Project A: Juan Pérez --firma--> Contrato A
        pA_juan = store.save_node(db_session, "proj-A", "Juan Pérez", "person", source_doc_id="doc-A1")
        pA_cont = store.save_node(db_session, "proj-A", "Contrato A", "contract", source_doc_id="doc-A2")
        store.save_edge(db_session, "proj-A", pA_juan.id, pA_cont.id, "firma", source_doc_id="doc-A-rel")
        store.update_build_status(db_session, "proj-A", "full")

        # Project B: Juan Pérez --trabaja_en--> Empresa B
        pB_juan = store.save_node(db_session, "proj-B", "Juan Pérez", "person", source_doc_id="doc-B1")
        pB_emp = store.save_node(db_session, "proj-B", "Empresa B", "org", source_doc_id="doc-B2")
        store.save_edge(db_session, "proj-B", pB_juan.id, pB_emp.id, "trabaja_en", source_doc_id="doc-B-rel")
        store.update_build_status(db_session, "proj-B", "full")

        # Query Project A
        qA = GraphQuery(project_id="proj-A", query="Juan Pérez", max_depth=2)
        resA = engine.query(qA, db=db_session)

        labelsA = {e.label for e in resA.entities}
        assert labelsA == {"Juan Pérez", "Contrato A"}
        assert "Empresa B" not in labelsA
        assert resA.relationships[0].relation_type == "firma"
        assert "doc-B1" not in resA.source_document_ids

        # Query Project B
        qB = GraphQuery(project_id="proj-B", query="Juan Pérez", max_depth=2)
        resB = engine.query(qB, db=db_session)

        labelsB = {e.label for e in resB.entities}
        assert labelsB == {"Juan Pérez", "Empresa B"}
        assert "Contrato A" not in labelsB
        assert resB.relationships[0].relation_type == "trabaja_en"
        assert "doc-A1" not in resB.source_document_ids


# ── 5. Determinism & Explainability Tests ───────────────────────


class TestDeterminismAndExplainability:
    def test_query_results_are_strictly_deterministic(self, db_session, engine, store):
        n1 = store.save_node(db_session, "proj-det", "Alpha", "concept")
        n2 = store.save_node(db_session, "proj-det", "Beta", "concept")
        n3 = store.save_node(db_session, "proj-det", "Gamma", "concept")

        store.save_edge(db_session, "proj-det", n1.id, n2.id, "rel_b")
        store.save_edge(db_session, "proj-det", n1.id, n3.id, "rel_a")
        store.update_build_status(db_session, "proj-det", "full")

        q = GraphQuery(project_id="proj-det", query="Alpha", max_depth=2)

        # Run multiple times
        res1 = engine.query(q, db=db_session)
        res2 = engine.query(q, db=db_session)

        # Entities, relationships and source_documents must be identical
        assert [e.model_dump() for e in res1.entities] == [e.model_dump() for e in res2.entities]
        assert [r.model_dump() for r in res1.relationships] == [r.model_dump() for r in res2.relationships]
        assert res1.source_document_ids == res2.source_document_ids
        assert res1.metadata["seeds_count"] == res2.metadata["seeds_count"]
        assert res1.metadata["traversal_depth"] == res2.metadata["traversal_depth"]

        # Verify ordering: entities sorted deterministically by label_normalized
        entity_labels = [e.label for e in res1.entities]
        assert entity_labels == ["Alpha", "Beta", "Gamma"]
        # Verify ordering: relationships sorted deterministically by relation_type
        rel_types = [r.relation_type for r in res1.relationships]
        assert rel_types == ["rel_a", "rel_b"]
