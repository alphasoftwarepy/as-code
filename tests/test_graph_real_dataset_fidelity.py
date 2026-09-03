"""
AS Core — Tests for Real Dataset + Project Isolation + Graph Fidelity (Gate 8.5)

Canonical Enterprise Dataset:
  - 01_empresa.md
  - 02_sistema.md
  - 03_proyecto.md
  - 04_politica.md
  - 05_documento_aislado.md

Verifies:
  1. Full Alpha dataset ingestion produces exact nodes and relationships.
  2. Cross-document entity resolution merges entities without duplicate records.
  3. Canonical expected relationships exist with exact provenance.
  4. Provenance tracking per document is strictly preserved on each edge.
  5. GraphQueryEngine responses for supervision (Query A).
  6. GraphQueryEngine responses for system usage (Query B).
  7. GraphQueryEngine responses for administration (Query C).
  8. GraphQueryEngine responses for project leadership (Query D).
  9. GraphQueryEngine responses for approval dependencies (Query E).
  10. Bounded multihop query respects depth and node limits.
  11. Beta dataset ingestion produces isolated Beta graph.
  12. Absolute isolation between Project Alpha and Project Beta.
  13. Homonymous entities across projects have distinct IDs and isolated scopes.
  14. Anti-false-relation: Carlos Benítez NEVER depends on María López.
  15. Complete list of prohibited/spurious relationships are confirmed absent.
  16. Full dataset ingestion does not alter RAG integrity or retrieval.
  17. Full dataset re-ingestion is strictly idempotent (SQL verified 0 duplicates).
  18. Multidocument entity (Carlos Benítez) connects across documents with preserved provenance.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.database import get_session, init_db
from api.graph_models import GraphEdge, GraphNode
from api.project_models import ProjectDocument
from api.rag_models import RAGDocument, RAGDocumentChunk
from api.rag_routes import rag_router
from runtime.graph.contracts import GraphQuery
from runtime.graph.ingestion import GraphIngestionPipeline
from runtime.graph.query import GraphQueryEngine
from runtime.graph.store import GraphStore
from runtime.projects.manager import ProjectManager


# ── Fixtures ───────────────────────────────────────────────────


DATASET_DIR = Path("as_core_graph_uix_tests")


@pytest.fixture
def fidelity_db(tmp_path):
    """Initializes an isolated temporary SQLite database."""
    db_file = str(tmp_path / "test_fidelity.db")
    init_db(db_file)
    db = get_session()
    yield db, db_file
    db.close()


@pytest.fixture
def pipeline():
    return GraphIngestionPipeline()


@pytest.fixture
def query_engine():
    return GraphQueryEngine()


@pytest.fixture
def populated_alpha_beta(fidelity_db, pipeline):
    """Pre-populates Project Alpha (01-04) and Project Beta (05)."""
    db, _ = fidelity_db
    pm = ProjectManager()

    proj_alpha = pm.create_project(db, name="Alpha Retail S.A", slug="alpha-retail")
    proj_beta = pm.create_project(db, name="Beta Services", slug="beta-services")

    # Ingest Alpha (01, 02, 03, 04)
    alpha_docs = ["01_empresa.md", "02_sistema.md", "03_proyecto.md", "04_politica.md"]
    for doc_name in alpha_docs:
        content = (DATASET_DIR / doc_name).read_text(encoding="utf-8")
        res = pipeline.ingest_document(db, proj_alpha.id, doc_name, content, commit=True)
        assert res.status == "success"

    # Ingest Beta (05)
    beta_doc = "05_documento_aislado.md"
    beta_content = (DATASET_DIR / beta_doc).read_text(encoding="utf-8")
    res_b = pipeline.ingest_document(db, proj_beta.id, beta_doc, beta_content, commit=True)
    assert res_b.status == "success"

    return proj_alpha.id, proj_beta.id


# ── TEST SUITE: Gate 8.5 ────────────────────────────────────────


class TestGraphRealDatasetFidelity:

    # ── 1. Full Dataset Ingestion & Topology ─────────────────────

    def test_full_alpha_dataset_ingestion(self, fidelity_db, populated_alpha_beta):
        """Verifica que el dataset Alpha completo (01-04) genera exactamente 8 nodos y 11 aristas."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        nodes = db.query(GraphNode).filter_by(project_id=alpha_id).all()
        edges = db.query(GraphEdge).filter_by(project_id=alpha_id).all()

        assert len(nodes) == 8
        assert len(edges) == 11

        labels = {n.label for n in nodes}
        expected_labels = {
            "Alpha Retail S.A",
            "María López",
            "Carlos Benítez",
            "Laura Gómez",
            "AR-01",
            "Migración 2026",
            "NovaSales",
            "Cambios de permisos",
        }
        assert labels == expected_labels

    def test_cross_document_entity_resolution(self, fidelity_db, populated_alpha_beta):
        """Demuestra que entidades mencionadas en múltiples documentos se resuelven a una única entidad."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        # Carlos Benítez aparece en 01, 02, 03, 04 -> debe haber exactamente 1 nodo en SQLite
        carlos_nodes = db.query(GraphNode).filter_by(project_id=alpha_id, label="Carlos Benítez").all()
        assert len(carlos_nodes) == 1
        carlos = carlos_nodes[0]
        assert sorted(carlos.source_doc_ids) == [
            "01_empresa.md",
            "02_sistema.md",
            "03_proyecto.md",
            "04_politica.md",
        ]

        # Laura Gómez aparece en 01, 02, 03, 04 -> 1 nodo único
        laura_nodes = db.query(GraphNode).filter_by(project_id=alpha_id, label="Laura Gómez").all()
        assert len(laura_nodes) == 1
        assert sorted(laura_nodes[0].source_doc_ids) == [
            "01_empresa.md",
            "02_sistema.md",
            "03_proyecto.md",
            "04_politica.md",
        ]

        # María López aparece en 01, 03, 04 -> 1 nodo único
        maria_nodes = db.query(GraphNode).filter_by(project_id=alpha_id, label="María López").all()
        assert len(maria_nodes) == 1
        assert sorted(maria_nodes[0].source_doc_ids) == [
            "01_empresa.md",
            "03_proyecto.md",
            "04_politica.md",
        ]

        # AR-01 aparece en 02, 03, 04 -> 1 nodo único
        ar01_nodes = db.query(GraphNode).filter_by(project_id=alpha_id, label="AR-01").all()
        assert len(ar01_nodes) == 1
        assert sorted(ar01_nodes[0].source_doc_ids) == [
            "02_sistema.md",
            "03_proyecto.md",
            "04_politica.md",
        ]

    def test_expected_alpha_relationships(self, fidelity_db, populated_alpha_beta):
        """Verifica que las 11 relaciones canónicas requeridas existen exactamente en Proyecto Alpha."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        nodes = db.query(GraphNode).filter_by(project_id=alpha_id).all()
        edges = db.query(GraphEdge).filter_by(project_id=alpha_id).all()
        node_map = {n.id: n.label for n in nodes}

        persisted_triplets = {
            (node_map[e.source_node_id], e.relation_type, node_map[e.target_node_id])
            for e in edges
        }

        expected_canonical = {
            ("Laura Gómez", "administra", "AR-01"),
            ("Laura Gómez", "administra permisos técnicos", "AR-01"),
            ("María López", "aprueba decisiones críticas", "Migración 2026"),
            ("Cambios de permisos", "deben ser aprobados por", "Laura Gómez"),
            ("Migración 2026", "depende de aprobación final", "María López"),
            ("Migración 2026", "depende de validación funcional", "Carlos Benítez"),
            ("Laura Gómez", "lidera", "Migración 2026"),
            ("Carlos Benítez", "participa como responsable funcional", "Migración 2026"),
            ("Carlos Benítez", "reporta", "María López"),
            ("María López", "supervisa", "Carlos Benítez"),
            ("Carlos Benítez", "utiliza", "AR-01"),
        }

        assert expected_canonical.issubset(persisted_triplets)
        assert len(persisted_triplets) == 11

    def test_relationship_provenance(self, fidelity_db, populated_alpha_beta):
        """Verifica que cada relación conserva su source_doc_id documental exacto."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        nodes = db.query(GraphNode).filter_by(project_id=alpha_id).all()
        edges = db.query(GraphEdge).filter_by(project_id=alpha_id).all()
        node_map = {n.id: n.label for n in nodes}

        provenance_map = {
            (node_map[e.source_node_id], e.relation_type, node_map[e.target_node_id]): e.source_doc_id
            for e in edges
        }

        # 01_empresa.md
        assert provenance_map[("María López", "supervisa", "Carlos Benítez")] == "01_empresa.md"
        assert provenance_map[("Carlos Benítez", "reporta", "María López")] == "01_empresa.md"

        # 02_sistema.md
        assert provenance_map[("Laura Gómez", "administra", "AR-01")] == "02_sistema.md"
        assert provenance_map[("Carlos Benítez", "utiliza", "AR-01")] == "02_sistema.md"

        # 03_proyecto.md
        assert provenance_map[("Laura Gómez", "lidera", "Migración 2026")] == "03_proyecto.md"
        assert provenance_map[("Carlos Benítez", "participa como responsable funcional", "Migración 2026")] == "03_proyecto.md"
        assert provenance_map[("Migración 2026", "depende de validación funcional", "Carlos Benítez")] == "03_proyecto.md"
        assert provenance_map[("Migración 2026", "depende de aprobación final", "María López")] == "03_proyecto.md"
        assert provenance_map[("María López", "aprueba decisiones críticas", "Migración 2026")] == "03_proyecto.md"

        # 04_politica.md
        assert provenance_map[("Laura Gómez", "administra permisos técnicos", "AR-01")] == "04_politica.md"
        assert provenance_map[("Cambios de permisos", "deben ser aprobados por", "Laura Gómez")] == "04_politica.md"

    # ── 2. Real Graph Queries (GraphQueryEngine) ────────────────

    def test_graph_query_supervision(self, fidelity_db, query_engine, populated_alpha_beta):
        """Consulta A: ¿Quién supervisa a Carlos Benítez? -> María López supervisa a Carlos Benítez."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        res = query_engine.query(
            GraphQuery(project_id=alpha_id, query="¿Quién supervisa a Carlos Benítez?"),
            db=db,
        )
        node_map = {e.id: e.label for e in res.entities}
        rels = [(node_map.get(r.source_id), r.relation_type, node_map.get(r.target_id)) for r in res.relationships]

        assert ("María López", "supervisa", "Carlos Benítez") in rels

    def test_graph_query_system_usage(self, fidelity_db, query_engine, populated_alpha_beta):
        """Consulta B: ¿Qué sistema utiliza Carlos Benítez? -> Carlos Benítez utiliza AR-01."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        res = query_engine.query(
            GraphQuery(project_id=alpha_id, query="¿Qué sistema utiliza Carlos Benítez?"),
            db=db,
        )
        node_map = {e.id: e.label for e in res.entities}
        rels = [(node_map.get(r.source_id), r.relation_type, node_map.get(r.target_id)) for r in res.relationships]

        assert ("Carlos Benítez", "utiliza", "AR-01") in rels

    def test_graph_query_system_administration(self, fidelity_db, query_engine, populated_alpha_beta):
        """Consulta C: ¿Quién administra AR-01? -> Laura Gómez administra AR-01."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        res = query_engine.query(
            GraphQuery(project_id=alpha_id, query="¿Quién administra AR-01?"),
            db=db,
        )
        node_map = {e.id: e.label for e in res.entities}
        rels = [(node_map.get(r.source_id), r.relation_type, node_map.get(r.target_id)) for r in res.relationships]

        assert ("Laura Gómez", "administra", "AR-01") in rels

    def test_graph_query_project_leadership(self, fidelity_db, query_engine, populated_alpha_beta):
        """Consulta D: ¿Quién lidera Migración 2026? -> Laura Gómez lidera Migración 2026."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        res = query_engine.query(
            GraphQuery(project_id=alpha_id, query="¿Quién lidera Migración 2026?"),
            db=db,
        )
        node_map = {e.id: e.label for e in res.entities}
        rels = [(node_map.get(r.source_id), r.relation_type, node_map.get(r.target_id)) for r in res.relationships]

        assert ("Laura Gómez", "lidera", "Migración 2026") in rels

    def test_graph_query_migration_dependencies(self, fidelity_db, query_engine, populated_alpha_beta):
        """Consulta E: ¿De quién depende la aprobación final de Migración 2026? -> María López."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        res = query_engine.query(
            GraphQuery(project_id=alpha_id, query="¿De quién depende la aprobación final de Migración 2026?"),
            db=db,
        )
        node_map = {e.id: e.label for e in res.entities}
        rels = [(node_map.get(r.source_id), r.relation_type, node_map.get(r.target_id)) for r in res.relationships]

        assert ("Migración 2026", "depende de aprobación final", "María López") in rels
        # NO existe 'Carlos Benítez depende de María López'
        assert ("Carlos Benítez", "depende de", "María López") not in rels

    def test_multihop_query_respects_bounds(self, fidelity_db, query_engine, populated_alpha_beta):
        """Consulta F Multihop: traversal respeta max_depth y max_nodes sin desbordamiento."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        res = query_engine.query(
            GraphQuery(
                project_id=alpha_id,
                query="Carlos Benítez",
                max_depth=2,
                max_nodes=5,
            ),
            db=db,
        )

        assert len(res.entities) <= 5
        assert len(res.entities) > 0

    # ── 3. Project Beta & Cross-Project Isolation ────────────────

    def test_beta_dataset_ingestion(self, fidelity_db, populated_alpha_beta):
        """Verifica que el dataset Beta (05_documento_aislado.md) se ingesta correctamente en Proyecto Beta."""
        db, _ = fidelity_db
        _, beta_id = populated_alpha_beta

        nodes = db.query(GraphNode).filter_by(project_id=beta_id).all()
        edges = db.query(GraphEdge).filter_by(project_id=beta_id).all()

        assert len(nodes) == 4
        assert len(edges) == 3

        labels = {n.label for n in nodes}
        assert labels == {"Beta Services", "BetaDesk", "Pedro Silva", "Ana Torres"}

        node_map = {n.id: n.label for n in nodes}
        triplets = {(node_map[e.source_node_id], e.relation_type, node_map[e.target_node_id]) for e in edges}

        assert ("Beta Services", "utiliza", "BetaDesk") in triplets
        assert ("Pedro Silva", "administra", "BetaDesk") in triplets
        assert ("Ana Torres", "supervisa", "Beta Services") in triplets

    def test_alpha_beta_project_isolation(self, fidelity_db, query_engine, populated_alpha_beta):
        """Aislamiento absoluto: Alpha jamás ve Beta y Beta jamás ve Alpha (en DB y en queries)."""
        db, _ = fidelity_db
        alpha_id, beta_id = populated_alpha_beta

        nodes_alpha = db.query(GraphNode).filter_by(project_id=alpha_id).all()
        nodes_beta = db.query(GraphNode).filter_by(project_id=beta_id).all()

        labels_alpha = {n.label for n in nodes_alpha}
        labels_beta = {n.label for n in nodes_beta}

        # Intersección vacía de entidades
        assert labels_alpha.isdisjoint(labels_beta)

        # Consultas desde Alpha NO retornan entidades de Beta
        res_a = query_engine.query(GraphQuery(project_id=alpha_id, query="BetaDesk Pedro Silva"), db=db)
        assert len(res_a.entities) == 0
        assert len(res_a.relationships) == 0

        # Consultas desde Beta NO retornan entidades de Alpha
        res_b = query_engine.query(GraphQuery(project_id=beta_id, query="AR-01 Carlos Benítez"), db=db)
        assert len(res_b.entities) == 0
        assert len(res_b.relationships) == 0

    def test_homonymous_entities_are_project_scoped(self, fidelity_db, pipeline, populated_alpha_beta):
        """Entidades homónimas en distintos proyectos tienen IDs distintos y ningún edge cruzado."""
        db, _ = fidelity_db
        alpha_id, beta_id = populated_alpha_beta

        # Ingestamos una persona homónima 'Carlos Benítez' en Proyecto Beta
        res = pipeline.ingest_document(
            db,
            beta_id,
            "doc_beta_carlos.txt",
            "Carlos Benítez supervisa las operaciones de BetaDesk.",
            commit=True,
        )
        assert res.status == "success"

        carlos_alpha = db.query(GraphNode).filter_by(project_id=alpha_id, label="Carlos Benítez").first()
        carlos_beta = db.query(GraphNode).filter_by(project_id=beta_id, label="Carlos Benítez").first()

        assert carlos_alpha is not None
        assert carlos_beta is not None
        assert carlos_alpha.id != carlos_beta.id
        assert carlos_alpha.project_id == alpha_id
        assert carlos_beta.project_id == beta_id

        # Verificar que ningún edge de Beta apunta a Alpha ni viceversa
        all_edges = db.query(GraphEdge).all()
        for edge in all_edges:
            src_node = db.query(GraphNode).filter_by(id=edge.source_node_id).first()
            tgt_node = db.query(GraphNode).filter_by(id=edge.target_node_id).first()
            assert src_node.project_id == edge.project_id
            assert tgt_node.project_id == edge.project_id

    # ── 4. Anti-False-Relation Invariants ───────────────────────

    def test_no_false_carlos_maria_dependency(self, fidelity_db, query_engine, populated_alpha_beta):
        """Prueba fundamental: Carlos Benítez NUNCA depende de María López."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        nodes = db.query(GraphNode).filter_by(project_id=alpha_id).all()
        edges = db.query(GraphEdge).filter_by(project_id=alpha_id).all()
        node_map = {n.id: n.label for n in nodes}

        triplets = [
            (node_map.get(e.source_node_id), e.relation_type, node_map.get(e.target_node_id))
            for e in edges
        ]

        # 1. No existe como edge directo
        assert not any(
            t[0] == "Carlos Benítez" and "depende" in t[1] and t[2] == "María López"
            for t in triplets
        )
        assert not any(
            t[0] == "María López" and "depende" in t[1] and t[2] == "Carlos Benítez"
            for t in triplets
        )

        # 2. No aparece en respuestas de GraphQueryEngine
        res = query_engine.query(
            GraphQuery(project_id=alpha_id, query="¿Carlos Benítez depende de María López?"),
            db=db,
        )
        query_triplets = [
            (node_map.get(r.source_id), r.relation_type, node_map.get(r.target_id))
            for r in res.relationships
        ]
        assert not any(
            t[0] == "Carlos Benítez" and "depende" in t[1] and t[2] == "María López"
            for t in query_triplets
        )

    def test_no_prohibited_relationships(self, fidelity_db, populated_alpha_beta):
        """Verifica exhaustivamente la ausencia del conjunto completo de relaciones espurias prohibidas."""
        db, _ = fidelity_db
        alpha_id, beta_id = populated_alpha_beta

        all_nodes = db.query(GraphNode).all()
        all_edges = db.query(GraphEdge).all()
        node_map = {n.id: n.label for n in all_nodes}

        triplets = [
            (node_map.get(e.source_node_id), e.relation_type, node_map.get(e.target_node_id))
            for e in all_edges
        ]

        prohibited_checks = [
            ("Carlos Benítez", "depende de", "María López"),
            ("María López", "depende de", "Carlos Benítez"),
            ("Carlos Benítez", "depende de", "Migración 2026"),
            ("María López", "utiliza", "AR-01"),
            ("Pedro Silva", "utiliza", "AR-01"),
            ("Laura Gómez", "supervisa", "Pedro Silva"),
            ("BetaDesk", "pertenece a", "Alpha Retail"),
            ("AR-01", "pertenece a", "Beta Services"),
        ]

        for src, rel, tgt in prohibited_checks:
            matching = [t for t in triplets if t[0] == src and rel in t[1] and t[2] == tgt]
            assert len(matching) == 0, f"Prohibited relation found in graph: {matching}"

    # ── 5. RAG Integrity & Idempotency ──────────────────────────

    def test_graph_does_not_corrupt_rag(self, fidelity_db):
        """Carga documental a través del endpoint real con RAG: RAG chunks y retrieval continúan funcionando."""
        db, _ = fidelity_db

        pm = ProjectManager()
        proj = pm.create_project(db, name="RAG Check Project", slug="rag-check-proj")
        chat = pm.create_chat(db, project_id=proj.id, session_id="sess-rag-check")

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

        # Upload 01_empresa.md
        content = (DATASET_DIR / "01_empresa.md").read_bytes()
        resp = client.post(
            f"/api/rag/documents/upload?pipeline=chat&session_id={chat.session_id}",
            files={"file": ("01_empresa.md", content, "text/markdown")},
        )
        assert resp.status_code == 200
        doc_id = resp.json()["document_id"]

        # RAG intacto
        rag_doc = db.query(RAGDocument).filter_by(id=doc_id).first()
        assert rag_doc is not None
        assert db.query(RAGDocumentChunk).filter_by(document_id=doc_id).count() == 1

        # Retrieval RAG funciona
        results = mock_rag.retrieve("María López", db)
        assert len(results) >= 1

        # Graph tables pobladas
        assert db.query(GraphNode).filter_by(project_id=proj.id).count() >= 2
        assert db.query(GraphEdge).filter_by(project_id=proj.id).count() >= 1

    def test_full_dataset_reingestion_is_idempotent(self, fidelity_db, pipeline, populated_alpha_beta):
        """Reingestión completa de los 5 documentos no produce duplicados en SQL."""
        db, _ = fidelity_db
        alpha_id, beta_id = populated_alpha_beta

        nodes_pre_alpha = db.query(GraphNode).filter_by(project_id=alpha_id).count()
        edges_pre_alpha = db.query(GraphEdge).filter_by(project_id=alpha_id).count()
        nodes_pre_beta = db.query(GraphNode).filter_by(project_id=beta_id).count()
        edges_pre_beta = db.query(GraphEdge).filter_by(project_id=beta_id).count()

        # Re-ingestión de todos los documentos en Alpha
        alpha_docs = ["01_empresa.md", "02_sistema.md", "03_proyecto.md", "04_politica.md"]
        for doc_name in alpha_docs:
            content = (DATASET_DIR / doc_name).read_text(encoding="utf-8")
            res = pipeline.ingest_document(db, alpha_id, doc_name, content, commit=True)
            assert res.status == "success"

        # Re-ingestión en Beta
        beta_content = (DATASET_DIR / "05_documento_aislado.md").read_text(encoding="utf-8")
        res_b = pipeline.ingest_document(db, beta_id, "05_documento_aislado.md", beta_content, commit=True)
        assert res_b.status == "success"

        # Conteos invariantes
        assert db.query(GraphNode).filter_by(project_id=alpha_id).count() == nodes_pre_alpha
        assert db.query(GraphEdge).filter_by(project_id=alpha_id).count() == edges_pre_alpha
        assert db.query(GraphNode).filter_by(project_id=beta_id).count() == nodes_pre_beta
        assert db.query(GraphEdge).filter_by(project_id=beta_id).count() == edges_pre_beta

        # Comprobación SQL directa de CERO duplicados en nodos
        dup_nodes = db.execute(
            text("""
            SELECT project_id, label_normalized, COUNT(*) as c
            FROM graph_nodes
            GROUP BY project_id, label_normalized
            HAVING c > 1
            """)
        ).fetchall()
        assert len(dup_nodes) == 0

        # Comprobación SQL directa de CERO duplicados en aristas
        dup_edges = db.execute(
            text("""
            SELECT project_id, source_node_id, target_node_id, relation_type, source_doc_id, COUNT(*) as c
            FROM graph_edges
            GROUP BY project_id, source_node_id, target_node_id, relation_type, source_doc_id
            HAVING c > 1
            """)
        ).fetchall()
        assert len(dup_edges) == 0

    def test_multidocument_entity_relationships(self, fidelity_db, populated_alpha_beta):
        """Demuestra que Carlos Benítez conecta con María López, AR-01 y Migración 2026 conservando provenance."""
        db, _ = fidelity_db
        alpha_id, _ = populated_alpha_beta

        nodes = db.query(GraphNode).filter_by(project_id=alpha_id).all()
        edges = db.query(GraphEdge).filter_by(project_id=alpha_id).all()
        node_map = {n.id: n.label for n in nodes}

        carlos_id = next(n.id for n in nodes if n.label == "Carlos Benítez")
        carlos_edges = [
            (node_map[e.source_node_id], e.relation_type, node_map[e.target_node_id], e.source_doc_id)
            for e in edges
            if e.source_node_id == carlos_id or e.target_node_id == carlos_id
        ]

        # Relaciones incidentes a Carlos Benítez
        triplets_only = [(t[0], t[1], t[2]) for t in carlos_edges]
        assert ("María López", "supervisa", "Carlos Benítez") in triplets_only
        assert ("Carlos Benítez", "reporta", "María López") in triplets_only
        assert ("Carlos Benítez", "utiliza", "AR-01") in triplets_only
        assert ("Carlos Benítez", "participa como responsable funcional", "Migración 2026") in triplets_only
        assert any(t[0] == "Migración 2026" and "depende de validación funcional" in t[1] and t[2] == "Carlos Benítez" for t in triplets_only)
