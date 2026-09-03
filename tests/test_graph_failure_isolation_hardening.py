"""
AS Core — Tests for Graph Failure Isolation and Idempotency Hardening (Gate 8.4)

Verifies:
  1. Extractor failure is isolated: RAG intact, no residue, no HTTP 500.
  2. Resolver failure is isolated: RAG intact, no partial residue.
  3. Node persistence failure triggers complete rollback: RAG intact.
  4. Partial failure (nodes created, edge fails) triggers complete rollback:
     no orphan nodes remain in database.
  5. RAG integrity verified: chunks, document content, and retrieval capability
     remain identical before and after Graph failure.
  6. Idempotency under repeated ingestion (10x): node and edge counts strictly stable.
  7. Direct SQL check for zero duplicate nodes (project_id, label_normalized).
  8. Direct SQL check for zero duplicate edges (project_id, src, tgt, rel, doc).
  9. Retry after failure succeeds: failed attempt 1 rolls back, retry attempt 2 commits cleanly.
  10. Retry after failure leaves no residue from the failed attempt.
  11. Partial failure with pre-existing graph: previous graph state is completely preserved
      while the failed document's partial nodes are rolled back.
  12. commit=False respects caller's transaction control.
  13. commit=True commits immediately on success.
  14. Project isolation holds under repeated retries across Project A and Project B.
  15. Documents without project skip Graph cleanly.
  16. Repeated attempts on documents without project create zero graph nodes/edges.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.database import get_session, init_db
from api.graph_models import GraphEdge, GraphNode
from api.project_models import ProjectDocument
from api.rag_models import RAGDocument, RAGDocumentChunk
from api.rag_routes import rag_router
from runtime.graph.extractor import StructuralExtractor
from runtime.graph.ingestion import GraphIngestionPipeline
from runtime.graph.resolver import EntityResolver
from runtime.graph.store import GraphStore
from runtime.projects.manager import ProjectManager


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def hardening_db(tmp_path):
    """Initializes an isolated temporary SQLite database."""
    db_file = str(tmp_path / "test_hardening.db")
    init_db(db_file)
    db = get_session()
    yield db, db_file
    db.close()


@pytest.fixture
def test_app(hardening_db):
    """FastAPI test app with rag_router and mock RAG service with retrieval."""
    db, _ = hardening_db
    app = FastAPI()
    app.include_router(rag_router)

    def mock_process_document(doc, session):
        chunk = RAGDocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            chunk_type="prose",
            text=doc.content[:150],
            meta_json='{"source": "test"}',
        )
        session.add(chunk)
        session.commit()
        return 1

    mock_rag = MagicMock()
    mock_rag.process_document.side_effect = mock_process_document

    # Simulate retrieval returning the chunk for RAG integrity testing
    def mock_retrieve(query, session, **kwargs):
        chunks = session.query(RAGDocumentChunk).all()
        return [c.text for c in chunks if query.lower() in c.text.lower()]

    mock_rag.retrieve.side_effect = mock_retrieve
    app.state.rag_service = mock_rag

    return app, mock_rag


# ── TEST SUITE: Gate 8.4 ────────────────────────────────────────


class TestGraphFailureIsolationHardening:

    # ── 1. Failure Isolation Scenarios ──────────────────────────

    def test_extractor_failure_isolated(self, test_app, hardening_db):
        """Caso A: Fallo del extractor no compromete RAG, no deja residuos y no produce HTTP 500."""
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Fail Extractor", slug="fail-ext")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-fail-ext")

        client = TestClient(app)
        doc_content = "María López supervisa a Carlos Benítez."

        with patch.object(StructuralExtractor, "extract", side_effect=RuntimeError("Extractor crashed")):
            response = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("ext_fail.txt", doc_content.encode("utf-8"), "text/plain")},
            )

        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # RAG intacto
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        assert rag_doc is not None
        chunks = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).all()
        assert len(chunks) == 1

        # Graph sin residuos
        nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        edges = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_resolver_failure_isolated(self, test_app, hardening_db):
        """Caso B: Fallo del resolver no compromete RAG y no deja residuos parciales."""
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Fail Resolver", slug="fail-res")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-fail-res")

        client = TestClient(app)
        doc_content = "Laura Gómez administra AR-01."

        with patch.object(EntityResolver, "ingest_extraction_result", side_effect=RuntimeError("Resolver crashed")):
            response = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("res_fail.txt", doc_content.encode("utf-8"), "text/plain")},
            )

        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # RAG intacto
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        assert rag_doc is not None
        chunks = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).all()
        assert len(chunks) == 1

        # Graph sin residuos
        nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        edges = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_node_persistence_failure_rolls_back(self, test_app, hardening_db):
        """Caso C: Excepción durante persistencia de nodo hace rollback y RAG queda intacto."""
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Fail Persist", slug="fail-persist")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-fail-persist")

        client = TestClient(app)
        doc_content = "Carlos Benítez utiliza AR-01."

        with patch.object(GraphStore, "save_node", side_effect=RuntimeError("Store save_node disk error")):
            response = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("persist_fail.txt", doc_content.encode("utf-8"), "text/plain")},
            )

        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # RAG intacto
        assert db.query(RAGDocument).filter(RAGDocument.id == doc_id).first() is not None
        assert db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).count() == 1

        # Graph sin residuos
        assert db.query(GraphNode).filter(GraphNode.project_id == proj.id).count() == 0
        assert db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).count() == 0

    def test_partial_graph_failure_rolls_back_completely(self, test_app, hardening_db):
        """
        Caso D (OBLIGATORIO):
        Fallo después de crear nodos pero antes de completar edges:
          node 1 -> creado en sesión
          node 2 -> creado en sesión
          edge 1 -> excepción forzada
        Verifica que el rollback elimina TODOS los nodos creados por la transacción fallida.
        No quedan nodos huérfanos.
        """
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Partial Fail", slug="partial-fail")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-partial-fail")

        client = TestClient(app)
        doc_content = "María López supervisa a Carlos Benítez."

        # Interceptamos ingest_document para crear nodos parcialmente en la sesión antes de lanzar excepción
        original_ingest = GraphIngestionPipeline.ingest_document

        def failing_partial_ingest(self, db_sess, project_id, document_id, text, **kwargs):
            # 1. Creamos 2 nodos parcialmente
            store = GraphStore()
            store.save_node(db_sess, project_id, "Nodo Parcial Uno", "person")
            store.save_node(db_sess, project_id, "Nodo Parcial Dos", "person")
            db_sess.flush()
            # 2. Simulamos fallo durante la creación del edge
            raise RuntimeError("Edge creation failure after nodes flushed")

        with patch.object(GraphIngestionPipeline, "ingest_document", failing_partial_ingest):
            response = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("partial_fail.txt", doc_content.encode("utf-8"), "text/plain")},
            )

        assert response.status_code == 200

        # Verificación directa en SQLite: CERO nodos huérfanos
        node_count = db.execute(
            text("SELECT COUNT(*) FROM graph_nodes WHERE project_id = :p"), {"p": proj.id}
        ).scalar()
        edge_count = db.execute(
            text("SELECT COUNT(*) FROM graph_edges WHERE project_id = :p"), {"p": proj.id}
        ).scalar()

        assert node_count == 0
        assert edge_count == 0

    # ── 2. RAG Integrity Verification ───────────────────────────

    def test_rag_intact_after_graph_failure(self, test_app, hardening_db):
        """Verifica que RAG (documento, chunks, contenido y retrieval) es 100% idéntico antes y después del fallo Graph."""
        app, mock_rag = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="RAG Integrity", slug="rag-intact")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-rag-intact")

        client = TestClient(app)
        doc_content = "Migración 2026 depende de la validación funcional de Carlos Benítez."

        # Simulamos fallo Graph
        with patch.object(GraphIngestionPipeline, "ingest_document", side_effect=RuntimeError("Graph down")):
            response = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("rag_check.txt", doc_content.encode("utf-8"), "text/plain")},
            )

        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # 1. Documento RAG
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        assert rag_doc is not None
        assert rag_doc.content == doc_content
        assert rag_doc.filename == "rag_check.txt"

        # 2. Chunks RAG
        chunks = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).all()
        assert len(chunks) == 1
        assert chunks[0].text == doc_content[:150]

        # 3. Retrieval RAG funciona sobre el contenido
        results = mock_rag.retrieve("Migración 2026", db)
        assert len(results) >= 1
        assert "Migración 2026" in results[0]

    # ── 3. Idempotency Hardening ────────────────────────────────

    def test_repeated_ingestion_is_idempotent_10x(self, test_app, hardening_db):
        """Procesar el mismo documento 10 veces consecutivas produce conteos estrictamente estables."""
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Repeat 10x", slug="repeat-10x")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-repeat-10x")

        client = TestClient(app)
        doc_content = """
        María López supervisa a Carlos Benítez.
        Carlos Benítez reporta a María López.
        Laura Gómez administra AR-01.
        Carlos Benítez utiliza AR-01.
        """

        # Ingestión inicial mediante upload
        resp = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("doc_10x.txt", doc_content.encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200
        doc_id = resp.json()["document_id"]

        nodes_initial = db.query(GraphNode).filter(GraphNode.project_id == proj.id).count()
        edges_initial = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).count()

        assert nodes_initial > 0
        assert edges_initial > 0

        # Repetir 9 veces más (total 10)
        pipeline = GraphIngestionPipeline()
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()

        for i in range(2, 11):
            res = pipeline.ingest_document(
                db=db,
                project_id=proj.id,
                document_id=doc_id,
                text=rag_doc.content,
                commit=True,
            )
            assert res.status == "success"

            nodes_current = db.query(GraphNode).filter(GraphNode.project_id == proj.id).count()
            edges_current = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).count()

            assert nodes_current == nodes_initial, f"Nodes drift on iteration {i}: {nodes_current} != {nodes_initial}"
            assert edges_current == edges_initial, f"Edges drift on iteration {i}: {edges_current} != {edges_initial}"

    def test_no_structural_duplicate_nodes_sql(self, test_app, hardening_db):
        """Consulta directa SQL: CERO grupos duplicados en (project_id, label_normalized)."""
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Dup SQL Nodes", slug="dup-nodes")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-dup-nodes")

        client = TestClient(app)
        doc_content = "María López supervisa a Carlos Benítez."

        # Ingestión triple
        for _ in range(3):
            client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("dup_test.txt", doc_content.encode("utf-8"), "text/plain")},
            )

        # Consulta SQL estricta de duplicados en nodos
        dup_nodes = db.execute(
            text("""
            SELECT project_id, label_normalized, COUNT(*) as c
            FROM graph_nodes
            WHERE project_id = :p
            GROUP BY project_id, label_normalized
            HAVING c > 1
            """),
            {"p": proj.id},
        ).fetchall()

        assert len(dup_nodes) == 0, f"Found duplicate nodes: {dup_nodes}"

    def test_no_structural_duplicate_edges_sql(self, test_app, hardening_db):
        """Consulta directa SQL: CERO grupos duplicados en (project_id, src, tgt, rel, doc)."""
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Dup SQL Edges", slug="dup-edges")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-dup-edges")

        client = TestClient(app)
        doc_content = "Laura Gómez administra AR-01."

        resp = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("dup_edges.txt", doc_content.encode("utf-8"), "text/plain")},
        )
        doc_id = resp.json()["document_id"]

        # Re-ingestión repetida del mismo doc
        pipeline = GraphIngestionPipeline()
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        for _ in range(5):
            pipeline.ingest_document(db, proj.id, doc_id, rag_doc.content, commit=True)

        # Consulta SQL estricta de duplicados en edges
        dup_edges = db.execute(
            text("""
            SELECT project_id, source_node_id, target_node_id, relation_type, source_doc_id, COUNT(*) as c
            FROM graph_edges
            WHERE project_id = :p
            GROUP BY project_id, source_node_id, target_node_id, relation_type, source_doc_id
            HAVING c > 1
            """),
            {"p": proj.id},
        ).fetchall()

        assert len(dup_edges) == 0, f"Found duplicate edges: {dup_edges}"

    # ── 4. Retry After Failure ──────────────────────────────────

    def test_retry_after_failure_succeeds_without_residue(self, test_app, hardening_db):
        """
        Intento 1: Graph falla -> rollback.
        Intento 2: Graph funciona -> commit.
        Verifica:
          - RAG permanece intacto.
          - Graph termina correctamente.
          - Cero residuos del intento 1.
          - Relaciones finales exactas.
        """
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Retry Test", slug="retry-test")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-retry")

        client = TestClient(app)
        doc_content = "María López supervisa a Carlos Benítez."

        # INTENTO 1: Fallo simulado
        with patch.object(GraphIngestionPipeline, "ingest_document", side_effect=RuntimeError("Transient DB glitch")):
            resp1 = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("retry_doc.txt", doc_content.encode("utf-8"), "text/plain")},
            )
            assert resp1.status_code == 200
            doc_id = resp1.json()["document_id"]

        # Comprobar que intento 1 dejó 0 residuos
        assert db.query(GraphNode).filter(GraphNode.project_id == proj.id).count() == 0
        assert db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).count() == 0

        # INTENTO 2: Retry limpio del documento existente
        pipeline = GraphIngestionPipeline()
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        res2 = pipeline.ingest_document(
            db=db,
            project_id=proj.id,
            document_id=doc_id,
            text=rag_doc.content,
            commit=True,
        )

        assert res2.status == "success"
        assert res2.nodes_created >= 2
        assert res2.edges_created == 1

        # Verificación final
        nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        edges = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()
        assert len(nodes) == 2
        assert len(edges) == 1

        node_map = {n.id: n.label for n in nodes}
        assert edges[0].relation_type == "supervisa"
        assert node_map[edges[0].source_node_id] == "María López"
        assert node_map[edges[0].target_node_id] == "Carlos Benítez"

    def test_partial_failure_preserves_preexisting_graph_state(self, test_app, hardening_db):
        """
        Un proyecto ya tiene datos de Doc 1.
        Doc 2 sufre un fallo parcial durante la ingestión.
        Verifica que el estado de Doc 1 queda intacto y Doc 2 no deja ningún nodo parcial.
        """
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Preexisting Graph", slug="preexist-graph")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-preexist")

        client = TestClient(app)

        # 1. Ingestión exitosa de Doc 1
        client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("doc_1.txt", "Laura Gómez administra AR-01.".encode("utf-8"), "text/plain")},
        )

        nodes_pre = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        edges_pre = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()
        count_nodes_pre = len(nodes_pre)
        count_edges_pre = len(edges_pre)
        assert count_nodes_pre >= 2
        assert count_edges_pre >= 1

        labels_pre = {n.label for n in nodes_pre}

        # 2. Ingestión de Doc 2 con fallo parcial
        def failing_ingest_partial(self, db_sess, project_id, document_id, text, **kwargs):
            store = GraphStore()
            # Escribe un nodo que debería ser revertido
            store.save_node(db_sess, project_id, "Carlos Benítez", "person")
            db_sess.flush()
            raise RuntimeError("Catastrophic error mid-ingestion of Doc 2")

        with patch.object(GraphIngestionPipeline, "ingest_document", failing_ingest_partial):
            client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("doc_2.txt", "Carlos Benítez utiliza AR-01.".encode("utf-8"), "text/plain")},
            )

        # 3. Comprobar que el estado es EXACTAMENTE el de Doc 1
        nodes_post = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        edges_post = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()

        assert len(nodes_post) == count_nodes_pre
        assert len(edges_post) == count_edges_pre
        assert {n.label for n in nodes_post} == labels_pre
        assert "Carlos Benítez" not in {n.label for n in nodes_post}

    # ── 5. Commit Control ───────────────────────────────────────

    def test_commit_false_respects_caller_transaction(self, hardening_db):
        """commit=False no confirma prematuramente la sesión y permite rollback total del llamador."""
        db, _ = hardening_db
        pipeline = GraphIngestionPipeline()
        project_id = "proj-tx-false"

        res = pipeline.ingest_document(
            db=db,
            project_id=project_id,
            document_id="doc-test-false",
            text="Laura Gómez administra AR-01.",
            commit=False,
        )
        assert res.status == "success"

        # El llamador ejecuta rollback
        db.rollback()

        # Ningún dato debió quedar confirmado
        store = GraphStore()
        assert len(store.list_nodes(db, project_id)) == 0
        assert len(store.list_edges(db, project_id)) == 0

    def test_commit_true_persists_success(self, hardening_db):
        """commit=True confirma la transacción: un rollback posterior no descarta los datos."""
        db, _ = hardening_db
        pipeline = GraphIngestionPipeline()
        project_id = "proj-tx-true"

        res = pipeline.ingest_document(
            db=db,
            project_id=project_id,
            document_id="doc-test-true",
            text="Laura Gómez administra AR-01.",
            commit=True,
        )
        assert res.status == "success"

        # Rollback posterior no descarta lo confirmado
        db.rollback()

        store = GraphStore()
        assert len(store.list_nodes(db, project_id)) >= 2
        assert len(store.list_edges(db, project_id)) == 1

    # ── 6. Project Isolation Under Retry ────────────────────────

    def test_project_isolation_under_retry(self, test_app, hardening_db):
        """
        Proyecto A (Carlos Benítez, AR-01) y Proyecto B (Carlos Benítez, BetaDesk).
        Múltiples retries de ingestión en A y B mantienen estricto aislamiento.
        """
        app, _ = test_app
        db, _ = hardening_db

        pm = ProjectManager()
        proj_a = pm.create_project(db, name="Project A", slug="proj-iso-a")
        chat_a = pm.create_chat(db, project_id=proj_a.id, session_id="sess-iso-a")

        proj_b = pm.create_project(db, name="Project B", slug="proj-iso-b")
        chat_b = pm.create_chat(db, project_id=proj_b.id, session_id="sess-iso-b")

        client = TestClient(app)

        # Ingestión A + retry A
        client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat_a.session_id}",
            files={"file": ("doc_a.txt", "Carlos Benítez utiliza AR-01.".encode("utf-8"), "text/plain")},
        )
        client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat_a.session_id}",
            files={"file": ("doc_a.txt", "Carlos Benítez utiliza AR-01.".encode("utf-8"), "text/plain")},
        )

        # Ingestión B + retry B
        client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat_b.session_id}",
            files={"file": ("doc_b.txt", "Carlos Benítez utiliza BetaDesk.".encode("utf-8"), "text/plain")},
        )
        client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat_b.session_id}",
            files={"file": ("doc_b.txt", "Carlos Benítez utiliza BetaDesk.".encode("utf-8"), "text/plain")},
        )

        nodes_a = db.query(GraphNode).filter(GraphNode.project_id == proj_a.id).all()
        nodes_b = db.query(GraphNode).filter(GraphNode.project_id == proj_b.id).all()

        labels_a = {n.label for n in nodes_a}
        labels_b = {n.label for n in nodes_b}

        assert "AR-01" in labels_a
        assert "BetaDesk" not in labels_a

        assert "BetaDesk" in labels_b
        assert "AR-01" not in labels_b

        carlos_a = next(n for n in nodes_a if n.label == "Carlos Benítez")
        carlos_b = next(n for n in nodes_b if n.label == "Carlos Benítez")
        assert carlos_a.id != carlos_b.id

    # ── 7. Documents Without Project ────────────────────────────

    def test_no_project_skips_graph(self, test_app, hardening_db):
        """Documento sin ProjectDocument: RAG exitoso, Graph omitido sin crear nodos globales."""
        app, _ = test_app
        db, _ = hardening_db

        client = TestClient(app)
        doc_content = "María López supervisa a Carlos Benítez."

        response = client.post(
            "/api/rag/documents/upload?pipeline=chat",
            files={"file": ("no_proj_doc.txt", doc_content.encode("utf-8"), "text/plain")},
        )

        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        assert db.query(RAGDocument).filter(RAGDocument.id == doc_id).first() is not None
        assert db.query(ProjectDocument).filter(ProjectDocument.document_id == doc_id).first() is None
        assert db.query(GraphNode).count() == 0
        assert db.query(GraphEdge).count() == 0

    def test_no_project_retry_creates_no_nodes(self, test_app, hardening_db):
        """Múltiples subidas sin proyecto mantienen las tablas de Graph absolutamente vacías."""
        app, _ = test_app
        db, _ = hardening_db

        client = TestClient(app)
        doc_content = "Carlos Benítez reporta a María López."

        for i in range(5):
            resp = client.post(
                "/api/rag/documents/upload?pipeline=chat",
                files={"file": (f"no_proj_{i}.txt", doc_content.encode("utf-8"), "text/plain")},
            )
            assert resp.status_code == 200

        # Verificación directa SQL
        total_nodes = db.execute(text("SELECT COUNT(*) FROM graph_nodes")).scalar()
        total_edges = db.execute(text("SELECT COUNT(*) FROM graph_edges")).scalar()

        assert total_nodes == 0
        assert total_edges == 0
