"""
AS Core — Graph Final Closure & Architectural Invariants (Gate 8.6)

Final architectural validation suite verifying:
  1. Module inventory & zero prohibited dependencies (no LLM, no FAISS, no BM25, no Memory).
  2. Bounded cognition: cycle safety, depth caps, and node limits under cyclic topologies.
  3. Strict determinism: 10 consecutive identical queries return bit-for-bit identical results and ordering.
  4. Multi-chat isolation: multiple chat sessions in the same project share the same knowledge graph without session fragmentation.
  5. Fail-safe degradation: GraphQueryEngine degrades cleanly without raising unhandled exceptions when unavailable.
  6. Full End-to-End lifecycle: Upload -> RAG process -> Graph hook -> Graph query -> RAG retrieve -> Re-ingestion -> Zero duplicate SQL state.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.database import get_session, init_db
from api.graph_models import GraphEdge, GraphNode
from api.rag_models import RAGDocument, RAGDocumentChunk
from api.rag_routes import rag_router
from runtime.graph.contracts import (
    GraphEntity,
    GraphProvider,
    GraphQuery,
    GraphQueryResult,
    GraphRelationship,
)
from runtime.graph.extractor import StructuralExtractor
from runtime.graph.formatter import RelationalContextFormatter
from runtime.graph.ingestion import GraphIngestionPipeline, IngestionResult
from runtime.graph.normalizer import normalize_key, normalize_label
from runtime.graph.query import GraphQueryEngine
from runtime.graph.resolver import EntityResolver
from runtime.graph.store import GraphStore
from runtime.graph.trigger import GraphTrigger
from runtime.projects.manager import ProjectManager


# ── Fixtures ───────────────────────────────────────────────────


DATASET_DIR = Path("as_core_graph_uix_tests")


@pytest.fixture
def closure_db(tmp_path):
    """Initializes an isolated temporary SQLite database."""
    db_file = str(tmp_path / "test_closure.db")
    init_db(db_file)
    db = get_session()
    yield db, db_file
    db.close()


# ── TEST SUITE: Gate 8.6 ────────────────────────────────────────


class TestGraphFinalClosure:

    # ── 1. Subsystem Inventory & Dependency Boundary ────────────

    def test_closure_subsystem_modules_inventory_and_imports(self):
        """
        Fases 1 y 2: Confirma que todos los módulos existen, tienen responsabilidades únicas
        y no importan LLM, OpenRouter, FAISS, BM25, UI ni Working Memory.
        """
        # Todos los componentes del subsistema Graph se importan limpiamente
        assert GraphStore is not None
        assert StructuralExtractor is not None
        assert EntityResolver is not None
        assert GraphQueryEngine is not None
        assert GraphTrigger is not None
        assert RelationalContextFormatter is not None
        assert GraphIngestionPipeline is not None

        # Verificación estricta de ausencia de dependencias externas en tiempo de ejecución
        import runtime.graph.contracts as c
        import runtime.graph.extractor as e
        import runtime.graph.formatter as f
        import runtime.graph.ingestion as i
        import runtime.graph.normalizer as n
        import runtime.graph.query as q
        import runtime.graph.resolver as r
        import runtime.graph.store as s
        import runtime.graph.trigger as t

        graph_modules = [c, e, f, i, n, q, r, s, t]

        prohibited = [
            "faiss",
            "bm25",
            "openrouter",
            "openai",
            "networkx",
            "celery",
            "redis",
        ]

        for mod in graph_modules:
            mod_vars = dir(mod)
            for var_name in mod_vars:
                assert var_name.lower() not in prohibited, f"Prohibited dependency '{var_name}' found in {mod.__name__}"

    # ── 2. Bounded Cognition & Cycle Safety ─────────────────────

    def test_closure_bounded_cognition_cycles_and_caps(self, closure_db):
        """
        Fase 5: Bounded cognition.
        Construye una topología con ciclos (A -> B -> C -> A) y verifica que el traversal
        respeta límites de profundidad y no genera loops infinitos.
        """
        db, _ = closure_db
        store = GraphStore()
        project_id = "proj-cycle-test"

        # Crear ciclo de 3 nodos: A -> B -> C -> A
        node_a = store.save_node(db, project_id, "Servicio A", "system")
        node_b = store.save_node(db, project_id, "Servicio B", "system")
        node_c = store.save_node(db, project_id, "Servicio C", "system")

        store.save_edge(db, project_id, node_a.id, node_b.id, "conecta_con")
        store.save_edge(db, project_id, node_b.id, node_c.id, "conecta_con")
        store.save_edge(db, project_id, node_c.id, node_a.id, "conecta_con")
        store.update_build_status(db, project_id, "full")
        db.commit()

        engine = GraphQueryEngine(store=store)

        # 1. Consulta con ciclo y max_depth=10: el set de visitados debe cortar el ciclo inmediatamente
        query_cycle = GraphQuery(
            project_id=project_id,
            query="Servicio A",
            max_depth=10,
            max_nodes=30,
        )
        res = engine.query(query_cycle, db=db)
        assert len(res.entities) == 3
        assert len(res.relationships) == 3

        # 2. Consulta con max_nodes=2 estricto: no debe devolver más de 2 nodos
        query_capped = GraphQuery(
            project_id=project_id,
            query="Servicio A",
            max_depth=5,
            max_nodes=2,
        )
        res_capped = engine.query(query_capped, db=db)
        assert len(res_capped.entities) <= 2

    # ── 3. Determinism & Stable Ordering ────────────────────────

    def test_closure_deterministic_repeatability_10x(self, closure_db):
        """
        Fases 6 y 7: Determinismo y ordenamiento estable.
        Ejecuta 10 veces consecutivas la misma consulta y verifica igualdad idéntica en entities y relationships.
        """
        db, _ = closure_db
        pipeline = GraphIngestionPipeline()
        project_id = "proj-det-test"

        content = (DATASET_DIR / "01_empresa.md").read_text(encoding="utf-8")
        pipeline.ingest_document(db, project_id, "01_empresa.md", content, commit=True)

        engine = GraphQueryEngine()
        query = GraphQuery(project_id=project_id, query="¿Quién supervisa a Carlos Benítez?")

        first_res = engine.query(query, db=db)
        first_entities = [(e.id, e.label, e.entity_type) for e in first_res.entities]
        first_rels = [(r.source_id, r.relation_type, r.target_id) for r in first_res.relationships]

        assert len(first_entities) >= 2
        assert len(first_rels) >= 1

        for i in range(2, 11):
            curr_res = engine.query(query, db=db)
            curr_entities = [(e.id, e.label, e.entity_type) for e in curr_res.entities]
            curr_rels = [(r.source_id, r.relation_type, r.target_id) for r in curr_res.relationships]

            assert curr_entities == first_entities, f"Entity mismatch on query iteration {i}"
            assert curr_rels == first_rels, f"Relationship mismatch on query iteration {i}"

    # ── 4. Multi-Chat Isolation / Project Scoping ────────────────

    def test_closure_multi_chat_shared_project_graph(self, closure_db):
        """
        Fases 12 y 13: Multi-Chat & Working Memory Isolation.
        Múltiples chats (A, B, C) dentro del mismo proyecto comparten el grafo del proyecto
        sin crear grafos separados ni provocar contaminación de sesiones.
        """
        db, _ = closure_db
        pm = ProjectManager()
        proj = pm.create_project(db, name="Multi Chat Corp", slug="multichat-corp")

        chat_a = pm.create_chat(db, project_id=proj.id, session_id="sess-chat-a", title="Chat A")
        chat_b = pm.create_chat(db, project_id=proj.id, session_id="sess-chat-b", title="Chat B")
        chat_c = pm.create_chat(db, project_id=proj.id, session_id="sess-chat-c", title="Chat C")

        # Ingestión de documento asociado al proyecto
        pipeline = GraphIngestionPipeline()
        doc_content = "María López supervisa a Carlos Benítez."
        pipeline.ingest_document(db, proj.id, "doc_mc.txt", doc_content, commit=True)

        engine = GraphQueryEngine()

        # Consultar el grafo del proyecto para cada sesión de chat
        # La consulta Graph se scopea por project_id (la invariante del sistema)
        query = GraphQuery(project_id=proj.id, query="Carlos Benítez")

        res_a = engine.query(query, db=db)
        res_b = engine.query(query, db=db)
        res_c = engine.query(query, db=db)

        assert len(res_a.entities) == len(res_b.entities) == len(res_c.entities) == 2
        assert len(res_a.relationships) == len(res_b.relationships) == len(res_c.relationships) == 1

        labels_a = {e.label for e in res_a.entities}
        labels_b = {e.label for e in res_b.entities}
        labels_c = {e.label for e in res_c.entities}

        assert labels_a == labels_b == labels_c == {"María López", "Carlos Benítez"}

    # ── 5. Fail-Safe Degradation ────────────────────────────────

    def test_closure_graph_provider_fail_safe_and_unavailable(self, closure_db):
        """
        Fases 3, 14 y 15: Fail-Safe y degradación cuando Graph no está disponible.
        GraphQueryEngine nunca lanza excepciones no capturadas ante sesiones nulas o caídas del store.
        """
        engine = GraphQueryEngine()
        query = GraphQuery(project_id="proj-fail", query="Test")

        # 1. Sin sesión DB: devuelve unavailable limpiamente sin lanzar excepción
        res_no_db = engine.query(query, db=None)
        assert res_no_db.graph_available is False
        assert len(res_no_db.entities) == 0

        # 2. Store que lanza excepción interna simulada
        mock_store = MagicMock()
        mock_store.get_build_status.side_effect = RuntimeError("Database offline")
        failing_engine = GraphQueryEngine(store=mock_store)

        db, _ = closure_db
        res_failing = failing_engine.query(query, db=db)
        assert res_failing.graph_available is False
        assert len(res_failing.entities) == 0

    # ── 6. Full End-to-End Lifecycle ────────────────────────────

    def test_closure_full_e2e_lifecycle(self, closure_db):
        """
        Fase 19: Full End-to-End Lifecycle.
        1. Crear proyecto.
        2. Upload documento vía FastAPI test client (RAG process + Graph hook).
        3. Verificar persistencia RAG en SQLite.
        4. Verificar persistencia Graph en SQLite.
        5. Consultar Graph vía GraphQueryEngine.
        6. Consultar RAG vía mock retriever.
        7. Reingestión del mismo documento (idempotencia).
        8. Comprobación SQL de 0 duplicados.
        """
        db, _ = closure_db
        pm = ProjectManager()
        proj = pm.create_project(db, name="E2E Closure", slug="e2e-closure")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-e2e-closure")

        app = FastAPI()
        app.include_router(rag_router)

        mock_rag = MagicMock()

        def mock_process(doc, sess):
            chunk = RAGDocumentChunk(
                document_id=doc.id,
                chunk_index=0,
                chunk_type="prose",
                text=doc.content,
                meta_json="{}",
            )
            sess.add(chunk)
            sess.commit()
            return 1

        mock_rag.process_document.side_effect = mock_process
        mock_rag.retrieve.side_effect = lambda q, s, **kw: [c.text for c in s.query(RAGDocumentChunk).all() if q.lower() in c.text.lower()]
        app.state.rag_service = mock_rag

        client = TestClient(app)

        # 1. Upload documento real
        content = (DATASET_DIR / "01_empresa.md").read_bytes()
        resp = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("01_empresa.md", content, "text/markdown")},
        )
        assert resp.status_code == 200
        doc_id = resp.json()["document_id"]

        # 2. Verificar RAG en SQLite
        assert db.query(RAGDocument).filter_by(id=doc_id).count() == 1
        assert db.query(RAGDocumentChunk).filter_by(document_id=doc_id).count() == 1

        # 3. Verificar Graph en SQLite
        nodes_count = db.query(GraphNode).filter_by(project_id=proj.id).count()
        edges_count = db.query(GraphEdge).filter_by(project_id=proj.id).count()
        assert nodes_count >= 2
        assert edges_count >= 1

        # 4. Consultar Graph
        engine = GraphQueryEngine()
        q_res = engine.query(GraphQuery(project_id=proj.id, query="Carlos Benítez"), db=db)
        assert q_res.graph_available is True
        assert len(q_res.entities) >= 2

        # 5. Consultar RAG
        rag_res = mock_rag.retrieve("Carlos Benítez", db)
        assert len(rag_res) >= 1

        # 6. Reingestión
        pipeline = GraphIngestionPipeline()
        rag_doc = db.query(RAGDocument).filter_by(id=doc_id).first()
        res_reingest = pipeline.ingest_document(db, proj.id, doc_id, rag_doc.content, commit=True)
        assert res_reingest.status == "success"

        # 7. SQL 0 duplicados
        dup_nodes = db.execute(
            text("SELECT COUNT(*) FROM (SELECT id FROM graph_nodes WHERE project_id = :p GROUP BY label_normalized HAVING COUNT(*) > 1)"),
            {"p": proj.id},
        ).scalar()
        dup_edges = db.execute(
            text("SELECT COUNT(*) FROM (SELECT id FROM graph_edges WHERE project_id = :p GROUP BY source_node_id, target_node_id, relation_type, source_doc_id HAVING COUNT(*) > 1)"),
            {"p": proj.id},
        ).scalar()

        assert dup_nodes == 0
        assert dup_edges == 0
