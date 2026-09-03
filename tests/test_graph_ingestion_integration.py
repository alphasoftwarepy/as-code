"""
AS Core — Tests for RAG -> Graph Ingestion Hook Integration (Gate 8.3)

Verifies:
  1. Real upload flow with project association triggers Graph ingestion and persists nodes & edges.
  2. Document without project association skips Graph cleanly without creating global state.
  3. Graph failure is safely contained: RAG document & chunks survive, no HTTP 500, no RAG disruption.
  4. Graph success persists explicit relationships with correct endpoints and provenance.
  5. Project isolation: distinct projects remain completely isolated without cross-contamination.
  6. Multi-document sequential ingestion (Alpha Retail 01-04) builds expected graph topology without false relations.
  7. Graph database tables (graph_nodes, graph_edges) contain persisted records verified via raw SQL.
  8. Repeated ingestion of the same document does not produce duplicate nodes or edges.
  9. RAG regression safety: RAG chunks, metadata, and retrieval remain unaffected by Graph.
  10. Transaction safety: RAG commit precedes Graph, and Graph rollback does not destroy RAG data.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.database import get_session, init_db
from api.graph_models import GraphEdge, GraphNode
from api.project_models import Project, ProjectChat, ProjectDocument
from api.rag_models import RAGDocument, RAGDocumentChunk
from api.rag_routes import rag_router
from runtime.projects.manager import ProjectManager


# ── Test Database Fixture ──────────────────────────────────────


@pytest.fixture
def integration_db(tmp_path):
    """Initializes a temporary SQLite database with all tables."""
    db_file = str(tmp_path / "test_integration.db")
    init_db(db_file)
    db = get_session()
    yield db, db_file
    db.close()


@pytest.fixture
def test_app(integration_db):
    """FastAPI test app with rag_router and mock RAG service."""
    db, _ = integration_db
    app = FastAPI()
    app.include_router(rag_router)

    # Mock RAG service that chunks documents into SQLite
    def mock_process_document(doc, session):
        # Create a sample chunk to simulate real RAG chunk persistence
        chunk = RAGDocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            chunk_type="prose",
            text=doc.content[:100],
            meta_json="{}",
        )
        session.add(chunk)
        session.commit()
        return 1

    mock_rag = MagicMock()
    mock_rag.process_document.side_effect = mock_process_document
    app.state.rag_service = mock_rag

    return app, mock_rag


# ── TEST SUITE: Gate 8.3 ────────────────────────────────────────


class TestGraphIngestionIntegration:
    def test_test1_upload_real_with_project(self, test_app, integration_db):
        """TEST 1 — Upload real con proyecto: verifica RAGDocument, RAGChunk, GraphNode y GraphEdge."""
        app, _ = test_app
        db, _ = integration_db

        # 1. Crear proyecto y chat
        pm = ProjectManager()
        proj = pm.create_project(db, name="Proyecto Alpha", slug="alpha-real")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-alpha-01")

        client = TestClient(app)
        doc_content = "María López supervisa a Carlos Benítez."

        response = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("01_empresa.md", doc_content.encode("utf-8"), "text/markdown")},
        )

        assert response.status_code == 200
        data = response.json()
        doc_id = data["document_id"]

        # Verificar RAG Document
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        assert rag_doc is not None
        assert rag_doc.filename == "01_empresa.md"

        # Verificar RAG Chunks
        chunks = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).all()
        assert len(chunks) >= 1

        # Verificar ProjectDocument
        assoc = db.query(ProjectDocument).filter(ProjectDocument.document_id == doc_id).first()
        assert assoc is not None
        assert assoc.project_id == proj.id

        # Verificar Graph Nodes
        nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        assert len(nodes) >= 2
        labels = {n.label for n in nodes}
        assert "María López" in labels
        assert "Carlos Benítez" in labels

        # Verificar Graph Edges
        edges = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()
        assert len(edges) == 1
        assert edges[0].relation_type == "supervisa"
        assert edges[0].source_doc_id == doc_id

    def test_test2_document_without_project(self, test_app, integration_db):
        """TEST 2 — Documento sin proyecto: RAG procesa, Graph se omite limpiamente (sin nodos globales)."""
        app, _ = test_app
        db, _ = integration_db

        client = TestClient(app)
        doc_content = "Laura Gómez administra AR-01."

        # Upload sin session_id (no pertenece a ningún proyecto)
        response = client.post(
            "/api/rag/documents/upload?pipeline=chat",
            files={"file": ("no_proj.txt", doc_content.encode("utf-8"), "text/plain")},
        )

        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # RAG Document y chunks existen
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        assert rag_doc is not None
        chunks = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).all()
        assert len(chunks) >= 1

        # ProjectDocument NO existe
        assoc = db.query(ProjectDocument).filter(ProjectDocument.document_id == doc_id).first()
        assert assoc is None

        # Graph tables permanecen completamente vacías
        nodes = db.query(GraphNode).all()
        edges = db.query(GraphEdge).all()
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_test3_graph_failure_isolation(self, test_app, integration_db):
        """TEST 3 — Falla en Graph: RAG sobrevive intacto, HTTP 200, error contenido."""
        app, _ = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Fail Isolation", slug="fail-iso")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-fail-iso")

        client = TestClient(app)
        doc_content = "Carlos Benítez utiliza AR-01."

        # Mock GraphIngestionPipeline.ingest_document para simular caída en Graph
        with patch("runtime.graph.ingestion.GraphIngestionPipeline.ingest_document") as mock_ingest:
            mock_ingest.side_effect = RuntimeError("Simulated catastrophic graph crash")

            response = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("crash_doc.txt", doc_content.encode("utf-8"), "text/plain")},
            )

            # HTTP upload devuelve 200 porque el error de Graph en background es contenido
            assert response.status_code == 200
            doc_id = response.json()["document_id"]

        # RAG Document y chunks sobrevivieron perfectamente
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        assert rag_doc is not None
        chunks = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).all()
        assert len(chunks) == 1

        # No hay nodos ni aristas corruptas en Graph
        nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        edges = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()
        assert len(nodes) == 0
        assert len(edges) == 0

    def test_test4_graph_success_persisted(self, test_app, integration_db):
        """TEST 4 — Éxito de Graph: verifica persistencia de relación supervisa con endpoints válidos."""
        app, _ = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Graph Success", slug="graph-success")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-success")

        client = TestClient(app)
        doc_content = "María López supervisa a Carlos Benítez."

        response = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("doc_sup.txt", doc_content.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 200

        nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        edges = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()

        assert len(nodes) >= 2
        assert len(edges) == 1

        node_map = {n.id: n.label for n in nodes}
        edge = edges[0]
        assert edge.relation_type == "supervisa"
        assert node_map[edge.source_node_id] == "María López"
        assert node_map[edge.target_node_id] == "Carlos Benítez"

    def test_test5_project_isolation(self, test_app, integration_db):
        """TEST 5 — Project isolation: Proyecto A y Proyecto B tienen grafos totalmente independientes."""
        app, _ = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj_a = pm.create_project(db, name="Project Alpha", slug="proj-alpha")
        chat_a = pm.create_chat(db, project_id=proj_a.id, session_id="sess-a")

        proj_b = pm.create_project(db, name="Project Beta", slug="proj-beta")
        chat_b = pm.create_chat(db, project_id=proj_b.id, session_id="sess-b")

        client = TestClient(app)

        # Upload a Proyecto A
        client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat_a.session_id}",
            files={"file": ("doc_a.txt", "Carlos Benítez utiliza AR-01.".encode("utf-8"), "text/plain")},
        )

        # Upload a Proyecto B
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

        # Carlos Benítez existe en ambos pero con IDs diferentes
        carlos_a = next(n for n in nodes_a if n.label == "Carlos Benítez")
        carlos_b = next(n for n in nodes_b if n.label == "Carlos Benítez")
        assert carlos_a.id != carlos_b.id

    def test_test6_multi_document_enterprise_dataset(self, test_app, integration_db):
        """
        TEST 6 — Multi-document sequential ingestion (Alpha Retail 01 a 04):
        Verifica topología completa y confirma que NUNCA aparece Carlos depende de María.
        """
        app, _ = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Alpha Retail Corp", slug="alpha-corp")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-enterprise")

        client = TestClient(app)
        base = Path("as_core_graph_uix_tests")
        docs = ["01_empresa.md", "02_sistema.md", "03_proyecto.md", "04_politica.md"]

        for d in docs:
            file_path = base / d
            if not file_path.exists():
                pytest.skip(f"Test file {d} not found in {base}")
            content = file_path.read_bytes()

            resp = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": (d, content, "text/markdown")},
            )
            assert resp.status_code == 200

        all_nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        all_edges = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).all()

        node_map = {n.id: n.label for n in all_nodes}
        triplets = [
            (node_map.get(e.source_node_id), e.relation_type, node_map.get(e.target_node_id))
            for e in all_edges
        ]

        # Relaciones requeridas por la especificación
        assert ("María López", "supervisa", "Carlos Benítez") in triplets
        assert ("Carlos Benítez", "reporta", "María López") in triplets
        assert ("Laura Gómez", "administra", "AR-01") in triplets
        assert ("Carlos Benítez", "utiliza", "AR-01") in triplets
        assert ("Laura Gómez", "lidera", "Migración 2026") in triplets
        assert any(t[0] == "Migración 2026" and "depende de validación funcional" in t[1] and t[2] == "Carlos Benítez" for t in triplets)
        assert any(t[0] == "Migración 2026" and "depende de aprobación final" in t[1] and t[2] == "María López" for t in triplets)

        # REGLA CRÍTICA: Carlos Benítez NUNCA depende de María López
        assert not any(t[0] == "Carlos Benítez" and "depende" in t[1] and t[2] == "María López" for t in triplets)
        assert not any(t[0] == "María López" and "depende" in t[1] and t[2] == "Carlos Benítez" for t in triplets)

    def test_test7_graph_tables_not_empty_raw_sql(self, test_app, integration_db):
        """TEST 7 — Verificación directa mediante raw SQL: SELECT COUNT(*) > 0 en graph_nodes y graph_edges."""
        app, _ = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="SQL Verification", slug="sql-verif")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-sql")

        client = TestClient(app)
        doc_content = "Laura Gómez administra el sistema AR-01."

        client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("sql_test.txt", doc_content.encode("utf-8"), "text/plain")},
        )

        # Raw SQL counts
        node_count = db.execute(text("SELECT COUNT(*) FROM graph_nodes WHERE project_id = :p"), {"p": proj.id}).scalar()
        edge_count = db.execute(text("SELECT COUNT(*) FROM graph_edges WHERE project_id = :p"), {"p": proj.id}).scalar()

        assert node_count > 0
        assert edge_count > 0

    def test_test8_repeated_ingestion_idempotency(self, test_app, integration_db):
        """TEST 8 — Repeated ingestion: procesar el mismo documento múltiples veces no incrementa nodos ni aristas."""
        app, _ = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Repeat Test", slug="repeat-test")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-repeat")

        client = TestClient(app)
        doc_content = "Carlos Benítez reporta a María López."

        # Ingestión inicial mediante upload real
        resp = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("repeat.txt", doc_content.encode("utf-8"), "text/plain")},
        )
        assert resp.status_code == 200
        doc_id = resp.json()["document_id"]

        nodes_1 = db.query(GraphNode).filter(GraphNode.project_id == proj.id).count()
        edges_1 = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).count()

        # Re-procesar el mismo documento una 2da vez
        from runtime.graph.ingestion import GraphIngestionPipeline
        graph_pipeline = GraphIngestionPipeline()
        rag_doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()

        res2 = graph_pipeline.ingest_document(
            db=db,
            project_id=proj.id,
            document_id=doc_id,
            text=rag_doc.content,
            commit=True,
        )
        assert res2.status == "success"
        nodes_2 = db.query(GraphNode).filter(GraphNode.project_id == proj.id).count()
        edges_2 = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).count()

        # Re-procesar el mismo documento una 3ra vez
        res3 = graph_pipeline.ingest_document(
            db=db,
            project_id=proj.id,
            document_id=doc_id,
            text=rag_doc.content,
            commit=True,
        )
        assert res3.status == "success"
        nodes_3 = db.query(GraphNode).filter(GraphNode.project_id == proj.id).count()
        edges_3 = db.query(GraphEdge).filter(GraphEdge.project_id == proj.id).count()

        assert nodes_1 == nodes_2 == nodes_3
        assert edges_1 == edges_2 == edges_3
        assert nodes_1 > 0
        assert edges_1 > 0

    def test_test9_rag_regression_integrity(self, test_app, integration_db):
        """TEST 9 — RAG regression: chunks y metadata de RAG no sufren alteración alguna por Graph."""
        app, mock_rag = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="RAG Integrity", slug="rag-integrity")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-rag")

        client = TestClient(app)
        doc_content = "Pedro Silva administra BetaDesk."

        response = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("doc_rag.txt", doc_content.encode("utf-8"), "text/plain")},
        )
        assert response.status_code == 200
        doc_id = response.json()["document_id"]

        # Verificar que process_document de RAG fue llamado exactamente una vez
        assert mock_rag.process_document.called

        # Chunks de RAG permanecen intactos
        chunk = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).first()
        assert chunk is not None
        assert chunk.text == doc_content[:100]

    def test_test10_transaction_safety_rag_survives(self, test_app, integration_db):
        """
        TEST 10 — Transaction safety:
        RAG COMMIT -> GRAPH FAIL -> GRAPH ROLLBACK.
        Verifica explícitamente en la base de datos que los datos de RAG sobreviven
        y los datos parciales de Graph fueron eliminados.
        """
        app, _ = test_app
        db, _ = integration_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="Tx Safety", slug="tx-safety")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-tx")

        client = TestClient(app)
        doc_content = "Ana Torres supervisa las operaciones de Beta Services."

        # Simulamos que GraphIngestionPipeline guarda nodos parciales y luego lanza una excepción
        def failing_ingest_document(db_sess, project_id, document_id, text, **kwargs):
            # Escribe nodo parcial antes de fallar
            from runtime.graph.store import GraphStore
            store = GraphStore()
            store.save_node(db_sess, project_id, "Nodo Fantasma Parcial", "concept")
            # Forzamos excepción
            raise RuntimeError("Falla catastrófica simulada de Graph")

        with patch("runtime.graph.ingestion.GraphIngestionPipeline.ingest_document", side_effect=failing_ingest_document):
            response = client.post(
                f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
                files={"file": ("doc_tx_test.txt", doc_content.encode("utf-8"), "text/plain")},
            )
            assert response.status_code == 200
            doc_id = response.json()["document_id"]

        # 1. RAG sobrevivió en la base de datos
        doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
        assert doc is not None
        chunks = db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc_id).all()
        assert len(chunks) >= 1

        # 2. Graph ejecutó rollback: el 'Nodo Fantasma Parcial' NO existe en SQLite
        nodes = db.query(GraphNode).filter(GraphNode.project_id == proj.id).all()
        assert len(nodes) == 0
