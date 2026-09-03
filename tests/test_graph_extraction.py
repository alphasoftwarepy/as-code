"""
AS Code — GATE 4: Extraction & Entity Resolution Test Suite

Validates:
  1. Normalization:
     - Case insensitivity, unicode NFC, stripping punctuation/whitespace.
     - Stripping diacritics and corporate legal suffixes.
  2. Structural Extraction:
     - Person, organization, contract, module, date extraction.
     - Relationship extraction with provenance (document_id).
     - Empty text or content without entities returns empty result safely.
  3. Conservative Entity Resolution:
     - Same label + same type merges idempotently.
     - Suffix / diacritic variants merge (e.g. 'Alpha S.A.' and 'Alpha').
     - Conflicting entity types with identical label DO NOT merge (false-split safety).
  4. Project Isolation in Resolution:
     - Identical entity labels in Project A and Project B resolve to distinct nodes.
     - Cross-project leakage is impossible.
  5. Ingestion Pipeline:
     - ExtractionResult end-to-end ingestion creates GraphNodes and GraphEdges.
     - Provenance is preserved on every node and edge.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.graph_models import GraphBase
from runtime.graph.extractor import ExtractedEntity, StructuralExtractor
from runtime.graph.normalizer import normalize_key, normalize_label, strip_accents
from runtime.graph.resolver import EntityResolver
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
def resolver(store):
    return EntityResolver(store=store)


@pytest.fixture
def extractor():
    return StructuralExtractor()


# ── Normalizer Tests ───────────────────────────────────────────


class TestNormalizer:
    def test_normalize_label_whitespace_and_case(self):
        assert normalize_label("  Juan   Pérez  ") == "juan pérez"
        assert normalize_label("«Empresa X»") == "empresa x"
        assert normalize_label('"Contrato 123"') == "contrato 123"

    def test_strip_accents(self):
        assert strip_accents("Pérez") == "Perez"
        assert strip_accents("Constitución") == "Constitucion"

    def test_normalize_key_corporate_suffix(self):
        assert normalize_key("Alpha Software S.A.") == "alpha software"
        assert normalize_key("Beta Corp Inc.") == "beta corp"
        assert normalize_key("Gamma SRL") == "gamma"


# ── Extractor Tests ────────────────────────────────────────────


class TestStructuralExtractor:
    def test_empty_content_returns_empty_result(self, extractor):
        res = extractor.extract("", document_id="doc-1")
        assert res.entities == []
        assert res.relationships == []
        assert res.document_id == "doc-1"

    def test_extract_contracts_and_persons(self, extractor):
        text = "En Buenos Aires, comparece: Sr. Juan Pérez, en el marco del Contrato N° 458-2026."
        res = extractor.extract(text, document_id="doc-legal")

        labels = [e.label for e in res.entities]
        types = [e.entity_type for e in res.entities]

        assert any("Juan Pérez" in lbl for lbl in labels)
        assert any("Contrato N° 458-2026" in lbl for lbl in labels)
        assert "person" in types
        assert "contract" in types

        # Check provenance
        for e in res.entities:
            assert e.document_id == "doc-legal"

    def test_extract_relationship_signature(self, extractor):
        text = "El firmante Sr. Carlos Gómez firma el Contrato de Locación en conformidad."
        res = extractor.extract(text, document_id="doc-cont")

        assert len(res.relationships) >= 1
        rel = res.relationships[0]
        assert "Carlos Gómez" in rel.source_label
        assert "Contrato de Locación" in rel.target_label
        assert rel.relation_type == "firma"
        assert rel.document_id == "doc-cont"

    def test_extract_code_modules(self, extractor):
        text = "import coordinator\nclass RuntimeCoordinator:\n    pass\ndef run_loop():\n    pass"
        res = extractor.extract(text, document_id="doc-py", domain="programming")

        labels = [e.label for e in res.entities]
        assert "coordinator" in labels
        assert "RuntimeCoordinator" in labels
        assert "run_loop" in labels


# ── Entity Resolver Tests ──────────────────────────────────────


class TestEntityResolver:
    def test_resolve_exact_label_merges(self, db_session, resolver, store):
        e1 = ExtractedEntity(label="Juan Pérez", entity_type="person", document_id="doc-1")
        e2 = ExtractedEntity(label="juan pérez", entity_type="person", document_id="doc-2")

        n1 = resolver.resolve_and_save_node(db_session, "proj-1", e1)
        n2 = resolver.resolve_and_save_node(db_session, "proj-1", e2)

        assert n1.id == n2.id
        nodes = store.list_nodes(db_session, "proj-1")
        assert len(nodes) == 1
        assert set(nodes[0].source_doc_ids) == {"doc-1", "doc-2"}

    def test_resolve_suffix_variant_merges(self, db_session, resolver, store):
        e1 = ExtractedEntity(label="Alpha Corp S.A.", entity_type="org", document_id="doc-1")
        e2 = ExtractedEntity(label="Alpha Corp", entity_type="org", document_id="doc-2")

        n1 = resolver.resolve_and_save_node(db_session, "proj-1", e1)
        n2 = resolver.resolve_and_save_node(db_session, "proj-1", e2)

        assert n1.id == n2.id
        nodes = store.list_nodes(db_session, "proj-1")
        assert len(nodes) == 1

    def test_type_mismatch_ambiguity_does_not_merge(self, db_session, resolver, store):
        # Case: A contract named 'Locación' and a concept/person with the same name
        e1 = ExtractedEntity(label="Contrato Alpha", entity_type="contract", document_id="doc-1")
        e2 = ExtractedEntity(label="Contrato Alpha", entity_type="person", document_id="doc-2")

        n1 = resolver.resolve_and_save_node(db_session, "proj-1", e1)
        n2 = resolver.resolve_and_save_node(db_session, "proj-1", e2)

        assert n1.id != n2.id
        nodes = store.list_nodes(db_session, "proj-1")
        assert len(nodes) == 2
        types = {n.entity_type for n in nodes}
        assert types == {"contract", "person"}

    def test_project_isolation_resolver(self, db_session, resolver, store):
        e1 = ExtractedEntity(label="Dr. Martin", entity_type="person", document_id="doc-1")
        e2 = ExtractedEntity(label="Dr. Martin", entity_type="person", document_id="doc-2")

        nA = resolver.resolve_and_save_node(db_session, "proj-A", e1)
        nB = resolver.resolve_and_save_node(db_session, "proj-B", e2)

        assert nA.id != nB.id
        assert store.list_nodes(db_session, "proj-A")[0].id == nA.id
        assert store.list_nodes(db_session, "proj-B")[0].id == nB.id

    def test_ingest_extraction_result_end_to_end(self, db_session, resolver, extractor, store):
        text = "El Sr. Roberto Gómez firma el Acuerdo N° 101 en la fecha 2026-03-01."
        extraction = extractor.extract(text, document_id="doc-agreement")

        node_count, edge_count = resolver.ingest_extraction_result(
            db=db_session,
            project_id="proj-e2e",
            extraction=extraction,
        )

        assert node_count >= 2
        assert edge_count >= 1

        nodes = store.list_nodes(db_session, "proj-e2e")
        edges = store.list_edges(db_session, "proj-e2e")

        assert len(nodes) == node_count
        assert len(edges) == edge_count

        # Provenance verified
        for n in nodes:
            assert "doc-agreement" in n.source_doc_ids
        for e in edges:
            assert e.source_doc_id == "doc-agreement"

        status = store.get_build_status(db_session, "proj-e2e")
        assert status.status == "full"


# ── GATE 8.1: Structural Extractor & Resolver Enhancements ─────


class TestGate81StructuralEnhancements:
    def test_test1_organizational_roles(self, extractor):
        """TEST 1 — Roles organizacionales: detecta personas en cargos estructurales."""
        text = """
        María López es la gerente general.
        Carlos Benítez es el jefe de ventas.
        Laura Gómez es responsable de tecnología.
        """
        res = extractor.extract(text, document_id="doc-roles")
        persons = {e.label: e for e in res.entities if e.entity_type == "person"}

        assert "María López" in persons
        assert "Carlos Benítez" in persons
        assert "Laura Gómez" in persons

        assert persons["María López"].metadata.get("role") == "gerente general"
        assert persons["Carlos Benítez"].metadata.get("role") == "jefe de ventas"
        assert "tecnología" in persons["Laura Gómez"].metadata.get("role", "")

    def test_test2_supervision_relationship(self, extractor):
        """TEST 2 — Supervisión: María López supervisa a Carlos Benítez."""
        text = "María López supervisa a Carlos Benítez."
        res = extractor.extract(text, document_id="doc-sup")

        assert len(res.relationships) == 1
        rel = res.relationships[0]
        assert rel.source_label == "María López"
        assert rel.relation_type == "supervisa"
        assert rel.target_label == "Carlos Benítez"
        assert rel.document_id == "doc-sup"

    def test_test3_reporting_relationship(self, extractor):
        """TEST 3 — Reporte: Carlos Benítez reporta a María López."""
        text = "Carlos Benítez reporta a María López."
        res = extractor.extract(text, document_id="doc-rep")

        assert len(res.relationships) == 1
        rel = res.relationships[0]
        assert rel.source_label == "Carlos Benítez"
        assert rel.relation_type == "reporta"
        assert rel.target_label == "María López"
        assert rel.document_id == "doc-rep"

    def test_test4_system_administration_and_usage(self, extractor):
        """TEST 4 — Sistema: Laura Gómez administra AR-01; Carlos Benítez utiliza AR-01."""
        text = """
        Laura Gómez administra AR-01.
        Carlos Benítez utiliza AR-01.
        """
        res = extractor.extract(text, document_id="doc-sys")

        rel_map = {(r.source_label, r.relation_type, r.target_label) for r in res.relationships}
        assert ("Laura Gómez", "administra", "AR-01") in rel_map
        assert ("Carlos Benítez", "utiliza", "AR-01") in rel_map

        system_labels = [e.label for e in res.entities if e.entity_type == "system"]
        assert "AR-01" in system_labels

    def test_test5_project_leadership_and_participation(self, extractor):
        """TEST 5 — Proyecto: Laura Gómez lidera Migración 2026; Carlos Benítez participa en Migración 2026."""
        text = """
        Laura Gómez lidera Migración 2026.
        Carlos Benítez participa como responsable funcional en Migración 2026.
        """
        res = extractor.extract(text, document_id="doc-proj")

        rel_map = {(r.source_label, r.relation_type, r.target_label) for r in res.relationships}
        assert ("Laura Gómez", "lidera", "Migración 2026") in rel_map
        assert ("Carlos Benítez", "participa como responsable funcional", "Migración 2026") in rel_map

    def test_test6_project_dependencies_strict_svo(self, extractor):
        """TEST 6 — Dependencias: conserva Sujeto -> Verbo -> Objeto y NO inventa Carlos depende de María."""
        text = """
        Migración 2026 depende de la validación funcional de Carlos Benítez.
        Migración 2026 depende de la aprobación final de María López.
        """
        res = extractor.extract(text, document_id="doc-deps")

        rel_tuples = [(r.source_label, r.relation_type, r.target_label) for r in res.relationships]

        assert any(
            r[0] == "Migración 2026" and "depende" in r[1] and r[2] == "Carlos Benítez"
            for r in rel_tuples
        )
        assert any(
            r[0] == "Migración 2026" and "depende" in r[1] and r[2] == "María López"
            for r in rel_tuples
        )

        # Regla crítica: Carlos Benítez NUNCA depende de María López
        assert not any(
            r[0] == "Carlos Benítez" and "depende" in r[1] and r[2] == "María López"
            for r in rel_tuples
        )
        assert not any(
            r[0] == "María López" and "depende" in r[1] and r[2] == "Carlos Benítez"
            for r in rel_tuples
        )

    def test_test7_cross_document_resolution(self, db_session, resolver, extractor, store):
        """TEST 7 — Cross-document resolution: Doc B conecta relación al nodo de Carlos creado en Doc A."""
        # Documento A crea Carlos Benítez
        text_A = "Carlos Benítez es el jefe de ventas."
        ext_A = extractor.extract(text_A, document_id="doc-A")
        n_A, e_A = resolver.ingest_extraction_result(db_session, "proj-cross", ext_A)
        assert n_A >= 1

        carlos_node = store.find_node_by_label(db_session, "proj-cross", "carlos benítez")
        assert carlos_node is not None
        carlos_id = carlos_node.id

        # Documento B contiene relación Carlos Benítez utiliza AR-01
        text_B = "Carlos Benítez utiliza AR-01."
        ext_B = extractor.extract(text_B, document_id="doc-B")
        n_B, e_B = resolver.ingest_extraction_result(db_session, "proj-cross", ext_B)
        assert e_B >= 1

        edges = store.list_edges(db_session, "proj-cross")
        utiliza_edge = next((e for e in edges if e.relation_type == "utiliza"), None)
        assert utiliza_edge is not None
        # Comprueba que el edge utiliza el ID del nodo preexistente creado en Doc A
        assert utiliza_edge.source_node_id == carlos_id

    def test_test8_project_isolation(self, db_session, resolver, extractor, store):
        """TEST 8 — Project isolation: Carlos Benítez en Proyecto A vs Proyecto B son nodos distintos."""
        text = "Carlos Benítez es el jefe de ventas."
        ext = extractor.extract(text, document_id="doc-iso")

        resolver.ingest_extraction_result(db_session, "proj-Alpha", ext)
        resolver.ingest_extraction_result(db_session, "proj-Beta", ext)

        node_A = store.find_node_by_label(db_session, "proj-Alpha", "carlos benítez")
        node_B = store.find_node_by_label(db_session, "proj-Beta", "carlos benítez")

        assert node_A is not None
        assert node_B is not None
        assert node_A.id != node_B.id
        assert node_A.project_id == "proj-Alpha"
        assert node_B.project_id == "proj-Beta"

    def test_test9_idempotent_ingestion(self, db_session, resolver, extractor, store):
        """TEST 9 — Idempotencia: Ingerir el mismo documento 3 veces no duplica nodos ni aristas."""
        text = """
        María López supervisa a Carlos Benítez.
        Carlos Benítez utiliza AR-01.
        """
        ext = extractor.extract(text, document_id="doc-idem")

        # Ingestión 1
        resolver.ingest_extraction_result(db_session, "proj-idem", ext)
        nodes_1 = len(store.list_nodes(db_session, "proj-idem"))
        edges_1 = len(store.list_edges(db_session, "proj-idem"))

        # Ingestión 2
        resolver.ingest_extraction_result(db_session, "proj-idem", ext)
        nodes_2 = len(store.list_nodes(db_session, "proj-idem"))
        edges_2 = len(store.list_edges(db_session, "proj-idem"))

        # Ingestión 3
        resolver.ingest_extraction_result(db_session, "proj-idem", ext)
        nodes_3 = len(store.list_nodes(db_session, "proj-idem"))
        edges_3 = len(store.list_edges(db_session, "proj-idem"))

        assert nodes_1 == nodes_2 == nodes_3
        assert edges_1 == edges_2 == edges_3

    def test_test10_special_anti_false_relation(self, extractor):
        """
        TEST ESPECIAL CONTRA FALSAS RELACIONES (Sección 19):
        Entrada:
          Migración 2026 depende de Carlos.
          Migración 2026 depende de María.
          Carlos reporta a María.
        Verificar que la ÚNICA relación entre Carlos y María es 'reporta'.
        NO debe existir ninguna relación de dependencia entre Carlos y María.
        """
        text = """
        Migración 2026 depende de Carlos.
        Migración 2026 depende de María.
        Carlos reporta a María.
        """
        res = extractor.extract(text, document_id="doc-special")

        carlos_maria_rels = [
            r for r in res.relationships
            if (r.source_label == "Carlos" and r.target_label == "María")
            or (r.source_label == "María" and r.target_label == "Carlos")
        ]

        assert len(carlos_maria_rels) == 1
        assert carlos_maria_rels[0].relation_type == "reporta"
        assert carlos_maria_rels[0].source_label == "Carlos"
        assert carlos_maria_rels[0].target_label == "María"

        # Comprobar ausencia absoluta de 'depende' entre Carlos y María
        assert not any(
            "depende" in r.relation_type
            for r in carlos_maria_rels
        )

    def test_enterprise_dataset_real_extraction(self, db_session, resolver, extractor, store):
        """Validación completa con los 4 documentos empresariales reales del proyecto Alpha Retail."""
        from pathlib import Path

        base = Path("as_core_graph_uix_tests")
        docs = ["01_empresa.md", "02_sistema.md", "03_proyecto.md", "04_politica.md"]

        for d in docs:
            file_path = base / d
            if not file_path.exists():
                pytest.skip(f"Test file {d} not found in {base}")
            text = file_path.read_text(encoding="utf-8")
            ext = extractor.extract(text, document_id=d)
            resolver.ingest_extraction_result(db_session, "proj-alpha-test", ext)

        all_nodes = store.list_nodes(db_session, "proj-alpha-test")
        all_edges = store.list_edges(db_session, "proj-alpha-test")

        # Nodos esperados
        labels = {n.label for n in all_nodes}
        assert "María López" in labels
        assert "Carlos Benítez" in labels
        assert "Laura Gómez" in labels
        assert "AR-01" in labels
        assert "Migración 2026" in labels
        assert "NovaSales" in labels

        # Aristas esperadas
        carlos = store.find_node_by_label(db_session, "proj-alpha-test", "Carlos Benítez")
        maria = store.find_node_by_label(db_session, "proj-alpha-test", "María López")

        carlos_maria_edges = [
            e for e in all_edges
            if (e.source_node_id == carlos.id and e.target_node_id == maria.id)
            or (e.source_node_id == maria.id and e.target_node_id == carlos.id)
        ]

        # Solo debe existir supervisa (María -> Carlos) y reporta (Carlos -> María)
        assert len(carlos_maria_edges) == 2
        rel_types = {e.relation_type for e in carlos_maria_edges}
        assert "supervisa" in rel_types
        assert "reporta" in rel_types
        assert not any("depende" in rt for rt in rel_types)


