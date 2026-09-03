"""
AS Core — Tests for Graph Ingestion Pipeline Adapter (Gate 8.2)

Verifies:
  1. Simple document ingestion with provenance and relationships.
  2. Controlled handling of empty or whitespace text.
  3. Controlled validation of invalid project_id.
  4. Controlled validation of invalid document_id.
  5. Multi-document / cross-document relationship resolution in same project.
  6. Project isolation across distinct project IDs.
  7. Idempotency across duplicate document ingestion runs.
  8. Atomic rollback on persistence failure preventing partial graph state.
  9. Structural extractor integrity: strict S-V-O preservation and no false relations.
  10. Transaction control: commit=False preserves caller ownership.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.graph_models import GraphBase, GraphEdge, GraphNode
from runtime.graph.extractor import StructuralExtractor
from runtime.graph.ingestion import GraphIngestionPipeline, IngestionResult
from runtime.graph.resolver import EntityResolver
from runtime.graph.store import GraphStore


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def db_session():
    """In-memory SQLite database session isolated per test."""
    engine = create_engine("sqlite:///:memory:")
    GraphBase.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def store():
    return GraphStore()


@pytest.fixture
def extractor():
    return StructuralExtractor()


@pytest.fixture
def resolver(store):
    return EntityResolver(store=store)


@pytest.fixture
def pipeline(extractor, resolver, store):
    return GraphIngestionPipeline(
        extractor=extractor,
        resolver=resolver,
        store=store,
    )


# ── TEST SUITE: Gate 8.2 ────────────────────────────────────────


class TestGraphIngestionPipelineAdapter:
    def test_test1_simple_ingestion(self, db_session, pipeline, store):
        """TEST 1 — Simple ingestion: persiste entidades, relación y provenance."""
        text = "María López supervisa a Carlos Benítez."
        project_id = "proj-simple-test"
        doc_id = "doc-01"

        result = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id=doc_id,
            text=text,
        )

        assert isinstance(result, IngestionResult)
        assert result.status == "success"
        assert result.nodes_created >= 2
        assert result.edges_created == 1
        assert result.document_id == doc_id
        assert result.project_id == project_id
        assert result.error is None

        # Verificar persistencia en store
        nodes = store.list_nodes(db_session, project_id)
        edges = store.list_edges(db_session, project_id)

        assert len(nodes) == 2
        labels = {n.label for n in nodes}
        assert "María López" in labels
        assert "Carlos Benítez" in labels

        # Verificar provenance en nodos
        for node in nodes:
            assert doc_id in node.source_doc_ids
            assert node.project_id == project_id

        # Verificar relación y su provenance
        assert len(edges) == 1
        edge = edges[0]
        assert edge.relation_type == "supervisa"
        assert edge.source_doc_id == doc_id
        assert edge.project_id == project_id

    def test_test2_empty_text(self, db_session, pipeline, store):
        """TEST 2 — Empty text: no produce error ni altera la base de datos."""
        project_id = "proj-empty-test"

        # Texto vacío
        res1 = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id="doc-empty",
            text="",
        )
        assert res1.status == "empty"
        assert res1.nodes_created == 0
        assert res1.edges_created == 0
        assert res1.error is None

        # Texto con solo espacios en blanco
        res2 = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id="doc-spaces",
            text="   \n\t  ",
        )
        assert res2.status == "empty"
        assert res2.nodes_created == 0
        assert res2.edges_created == 0

        # Base de datos permanece limpia
        assert len(store.list_nodes(db_session, project_id)) == 0
        assert len(store.list_edges(db_session, project_id)) == 0

    def test_test3_invalid_project(self, db_session, pipeline, store):
        """TEST 3 — Invalid project: project_id vacío o inválido produce fallo controlado."""
        res = pipeline.ingest_document(
            db=db_session,
            project_id="",
            document_id="doc-valid",
            text="María López supervisa a Carlos Benítez.",
        )

        assert res.status == "failed"
        assert "project_id" in res.error.lower()
        assert res.nodes_created == 0
        assert res.edges_created == 0

        # Ningún nodo creado
        assert len(store.list_nodes(db_session, "")) == 0

    def test_test4_invalid_document(self, db_session, pipeline, store):
        """TEST 4 — Invalid document: document_id vacío produce fallo controlado."""
        res = pipeline.ingest_document(
            db=db_session,
            project_id="proj-valid",
            document_id="",
            text="María López supervisa a Carlos Benítez.",
        )

        assert res.status == "failed"
        assert "document_id" in res.error.lower()
        assert res.nodes_created == 0
        assert res.edges_created == 0

        assert len(store.list_nodes(db_session, "proj-valid")) == 0

    def test_test5_cross_document_resolution(self, db_session, pipeline, store):
        """TEST 5 — Cross-document: Doc B conecta relación al nodo creado en Doc A."""
        project_id = "proj-cross-doc"

        # Ingestión Doc A: define a Carlos Benítez
        res_a = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id="doc-A",
            text="Carlos Benítez es el jefe de ventas.",
        )
        assert res_a.status == "success"
        carlos_node = store.find_node_by_label(db_session, project_id, "Carlos Benítez")
        assert carlos_node is not None
        carlos_id = carlos_node.id
        assert "doc-A" in carlos_node.source_doc_ids

        # Ingestión Doc B: relación Carlos Benítez utiliza AR-01
        res_b = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id="doc-B",
            text="Carlos Benítez utiliza AR-01.",
        )
        assert res_b.status == "success"
        assert res_b.edges_created >= 1

        # Verificar que el edge conecta al Carlos creado en Doc A
        edges = store.list_edges(db_session, project_id)
        utiliza_edge = next((e for e in edges if e.relation_type == "utiliza"), None)
        assert utiliza_edge is not None
        assert utiliza_edge.source_node_id == carlos_id
        assert utiliza_edge.source_doc_id == "doc-B"

        # Verificar que el nodo de Carlos ahora tiene trazabilidad de ambos documentos
        carlos_updated = store.find_node_by_label(db_session, project_id, "Carlos Benítez")
        assert "doc-A" in carlos_updated.source_doc_ids
        assert "doc-B" in carlos_updated.source_doc_ids

    def test_test6_project_isolation(self, db_session, pipeline, store):
        """TEST 6 — Project isolation: entidades homónimas en proyectos distintos son independientes."""
        text = "Carlos Benítez es el jefe de ventas."

        res_a = pipeline.ingest_document(
            db=db_session,
            project_id="project-Alpha",
            document_id="doc-alpha",
            text=text,
        )
        res_b = pipeline.ingest_document(
            db=db_session,
            project_id="project-Beta",
            document_id="doc-beta",
            text=text,
        )

        assert res_a.status == "success"
        assert res_b.status == "success"

        node_a = store.find_node_by_label(db_session, "project-Alpha", "Carlos Benítez")
        node_b = store.find_node_by_label(db_session, "project-Beta", "Carlos Benítez")

        assert node_a is not None
        assert node_b is not None
        assert node_a.id != node_b.id
        assert node_a.project_id == "project-Alpha"
        assert node_b.project_id == "project-Beta"

    def test_test7_idempotency(self, db_session, pipeline, store):
        """TEST 7 — Idempotency: procesar el mismo documento tres veces no duplica nodos ni aristas."""
        text = """
        María López supervisa a Carlos Benítez.
        Carlos Benítez utiliza AR-01.
        """
        project_id = "proj-idempotency"
        doc_id = "doc-repeat"

        res1 = pipeline.ingest_document(db=db_session, project_id=project_id, document_id=doc_id, text=text)
        nodes_1 = len(store.list_nodes(db_session, project_id))
        edges_1 = len(store.list_edges(db_session, project_id))

        res2 = pipeline.ingest_document(db=db_session, project_id=project_id, document_id=doc_id, text=text)
        nodes_2 = len(store.list_nodes(db_session, project_id))
        edges_2 = len(store.list_edges(db_session, project_id))

        res3 = pipeline.ingest_document(db=db_session, project_id=project_id, document_id=doc_id, text=text)
        nodes_3 = len(store.list_nodes(db_session, project_id))
        edges_3 = len(store.list_edges(db_session, project_id))

        assert res1.status == "success"
        assert res2.status == "success"
        assert res3.status == "success"

        assert nodes_1 == nodes_2 == nodes_3
        assert edges_1 == edges_2 == edges_3

    def test_test8_atomic_rollback(self, db_session, store, extractor):
        """TEST 8 — Rollback: una excepción durante la persistencia ejecuta rollback y no deja estado parcial."""
        project_id = "proj-rollback-test"

        # Resolver mock que simula falla tras escribir parcialmente un nodo
        class FailingResolver(EntityResolver):
            def ingest_extraction_result(self, db, project_id, extraction):
                # Guarda un nodo parcial directamente en la sesión
                self.store.save_node(db, project_id, "Nodo Parcial Invalido", "concept")
                # Lanza excepción simulada
                raise RuntimeError("Controlled persistence error in Graph ingestion")

        failing_pipeline = GraphIngestionPipeline(
            extractor=extractor,
            resolver=FailingResolver(store=store),
            store=store,
        )

        # Con propagate_exceptions=True (por defecto)
        with pytest.raises(RuntimeError, match="Controlled persistence error"):
            failing_pipeline.ingest_document(
                db=db_session,
                project_id=project_id,
                document_id="doc-fail",
                text="María López supervisa a Carlos Benítez.",
                propagate_exceptions=True,
            )

        # Verificar que el rollback descartó todo estado parcial
        nodes_remaining = store.list_nodes(db_session, project_id)
        edges_remaining = store.list_edges(db_session, project_id)
        assert len(nodes_remaining) == 0
        assert len(edges_remaining) == 0

        # Con propagate_exceptions=False: resultado controlado con status="failed"
        res_controlled = failing_pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id="doc-fail-2",
            text="María López supervisa a Carlos Benítez.",
            propagate_exceptions=False,
        )
        assert res_controlled.status == "failed"
        assert "Controlled persistence error" in res_controlled.error
        assert len(store.list_nodes(db_session, project_id)) == 0

    def test_test9_extractor_integrity_and_anti_false_relation(self, db_session, pipeline, store):
        """
        TEST DE INTEGRIDAD DEL EXTRACTOR (Sección 18):
        Verifica que el adapter persista exactamente las relaciones explícitas:
          Migración 2026 -> depende de validación funcional -> Carlos Benítez
          Migración 2026 -> depende de aprobación final -> María López
          Carlos Benítez -> reporta -> María López
        Y que NO aparezca:
          Carlos Benítez -> depende de -> María López
        """
        text = """
        Migración 2026 depende de la validación funcional de Carlos Benítez.
        Migración 2026 depende de la aprobación final de María López.
        Carlos Benítez reporta a María López.
        """
        project_id = "proj-integrity-test"
        doc_id = "doc-integrity"

        result = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id=doc_id,
            text=text,
        )

        assert result.status == "success"

        all_nodes = store.list_nodes(db_session, project_id)
        all_edges = store.list_edges(db_session, project_id)

        node_map = {n.id: n.label for n in all_nodes}
        edge_triplets = [
            (node_map.get(e.source_node_id), e.relation_type, node_map.get(e.target_node_id))
            for e in all_edges
        ]

        # Relaciones requeridas explícitamente
        assert any(
            t[0] == "Migración 2026" and "depende de validación funcional" in t[1] and t[2] == "Carlos Benítez"
            for t in edge_triplets
        )
        assert any(
            t[0] == "Migración 2026" and "depende de aprobación final" in t[1] and t[2] == "María López"
            for t in edge_triplets
        )
        assert ("Carlos Benítez", "reporta", "María López") in edge_triplets

        # Regla de oro: NO existe 'Carlos Benítez depende de María López'
        assert not any(
            t[0] == "Carlos Benítez" and "depende" in t[1] and t[2] == "María López"
            for t in edge_triplets
        )
        assert not any(
            t[0] == "María López" and "depende" in t[1] and t[2] == "Carlos Benítez"
            for t in edge_triplets
        )

    def test_test10_transaction_commit_control(self, db_session, pipeline, store):
        """TEST 10 — Transaction control: commit=False no confirma prematuramente la sesión."""
        text = "Laura Gómez administra AR-01."
        project_id = "proj-tx-control"

        # commit=False deja la sesión abierta bajo control del caller
        res = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id="doc-tx",
            text=text,
            commit=False,
        )
        assert res.status == "success"

        # El caller puede hacer rollback si lo desea
        db_session.rollback()
        assert len(store.list_nodes(db_session, project_id)) == 0

        # Con commit=True, el adapter confirma explícitamente
        res_committed = pipeline.ingest_document(
            db=db_session,
            project_id=project_id,
            document_id="doc-tx-2",
            text=text,
            commit=True,
        )
        assert res_committed.status == "success"

        # Un rollback posterior no descarta lo ya confirmado
        db_session.rollback()
        assert len(store.list_nodes(db_session, project_id)) == 2
