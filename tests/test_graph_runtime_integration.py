"""
AS Code — GATE 7: Graph Runtime Integration Test Suite

Validates the optional, lazy, bounded, project-scoped, and fail-safe integration
of the Graph Layer into the AS-Core runtime (PureCoordinator.assemble).

Tests:
  1. RAG-centric query:
     - Informational query -> Graph OFF, graph_used=False, normal runtime flow.
  2. Relational query:
     - Relational query with seeded Graph -> graph_enabled=True, graph_used=True,
       "## RELATIONAL CONTEXT" present in system prompt snapshot.
  3. Graph provider failure (Fail-safe):
     - Exception in GraphProvider.query() -> assemble() succeeds, RAG intact,
       graph_used=False.
  4. No ProjectChat (Project isolation):
     - Session without ProjectChat association -> project_id=None, graph_used=False,
       graph_activation_reason="No project associated with active session".
  5. graph_provider=None (Backward compatibility):
     - Coordinator called with graph_provider=None -> graph_enabled=False,
       graph_used=False, output identical to pre-Gate 7.
  6. Char budget enforcement:
     - System prompt near budget limit -> remaining budget respected,
       len(system_prompt_snapshot) <= char_budget, no overflow.
  7. Strict GraphProvider contract compatibility:
     - Provider implementing strictly query(graph_query) without 'db' param
       operates without TypeError.
  8. Concrete GraphQueryEngine compatibility:
     - Concrete GraphQueryEngine requiring db session receives db and executes
       traversal successfully.
"""

import time
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.rag_models import Base
from api.graph_models import GraphBase
from api.memory_models import MemoryBase, MemoryVariable
from api.project_models import Project, ProjectBase, ProjectChat
from runtime.coordinator.manager import PureCoordinator
from runtime.coordinator.models import RuntimeContract, SessionSnapshot, WorkflowState
from runtime.graph.contracts import (
    GraphEntity,
    GraphProvider,
    GraphQuery,
    GraphQueryResult,
    GraphRelationship,
)
from runtime.graph.query import GraphQueryEngine
from runtime.graph.store import GraphStore


# ── Database Fixture ─────────────────────────────────────────────


@pytest.fixture
def db_session():
    """In-memory SQLite database containing all schemas for runtime integration."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    MemoryBase.metadata.create_all(bind=engine)
    ProjectBase.metadata.create_all(bind=engine)
    GraphBase.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_graph(db_session):
    """Seed project, chat, and graph store with relational data."""
    store = GraphStore()
    project_id = "proj-acme-1"
    session_id = "sess-test-1"

    # 1. Create Project and ProjectChat
    proj = Project(
        id=project_id,
        name="Acme Legal",
        slug="acme-legal",
        description="Proyecto de prueba",
    )
    db_session.add(proj)

    chat = ProjectChat(
        id="chat-1",
        session_id=session_id,
        project_id=project_id,
        title="Chat Acme",
    )
    db_session.add(chat)
    db_session.commit()

    # 2. Seed Graph Nodes
    n1 = store.save_node(
        db_session,
        project_id,
        "Carlos Pérez",
        "person",
        source_doc_id="doc-contrato-1",
    )
    n2 = store.save_node(
        db_session,
        project_id,
        "TechCorp Inc",
        "org",
        source_doc_id="doc-contrato-1",
    )
    n3 = store.save_node(
        db_session,
        project_id,
        "Contrato Marco",
        "contract",
        source_doc_id="doc-contrato-1",
    )

    # 3. Seed Graph Edges
    store.save_edge(
        db_session,
        project_id,
        n1.id,
        n3.id,
        "firmó",
        source_doc_id="doc-contrato-1",
    )
    store.save_edge(
        db_session,
        project_id,
        n3.id,
        n2.id,
        "vincula_con",
        source_doc_id="doc-contrato-1",
    )

    # 4. Mark graph as ready
    store.update_build_status(db_session, project_id, "entities_and_relations")

    engine = GraphQueryEngine(store=store)
    return {
        "project_id": project_id,
        "session_id": session_id,
        "store": store,
        "engine": engine,
    }


def _make_contract(session_id: str, user_message: str) -> RuntimeContract:
    """Helper to create minimal valid RuntimeContract."""
    snapshot = SessionSnapshot(
        session_id=session_id,
        turn_number=1,
        rag_query_stack=[user_message],
        language_history=[("ES", 1)],
    )
    return RuntimeContract(
        request_id=f"req-{int(time.time()*1000)}",
        session_id=session_id,
        model_id="gemma-chat",
        user_message=user_message,
        timestamp=time.time(),
        snapshot=snapshot,
    )


# ── Tests ────────────────────────────────────────────────────────


class TestGraphRuntimeIntegration:
    def test_1_rag_centric_query_graph_off(self, db_session, seeded_graph):
        """
        TEST 1: Informational/content query naturally served by RAG.
        Trigger evaluates needed=False -> Graph OFF, graph_used=False.
        """
        coord = PureCoordinator()
        contract = _make_contract(
            session_id=seeded_graph["session_id"],
            user_message="¿Qué dice la cláusula 5 de este contrato?",
        )

        manifest = coord.assemble(
            db=db_session,
            contract=contract,
            graph_provider=seeded_graph["engine"],
        )

        assert manifest.graph_enabled is True
        assert manifest.graph_used is False
        assert manifest.graph_entities_count == 0
        assert manifest.graph_relationships_count == 0
        assert "## RELATIONAL CONTEXT" not in manifest.system_prompt_snapshot

    def test_2_relational_query_graph_on(self, db_session, seeded_graph):
        """
        TEST 2: Relational query with seeded Graph.
        Trigger evaluates needed=True -> Graph ON, graph_used=True,
        relational context injected into system prompt snapshot.
        """
        coord = PureCoordinator()
        contract = _make_contract(
            session_id=seeded_graph["session_id"],
            user_message="¿Cómo se relaciona Carlos Pérez con TechCorp Inc a través de contratos?",
        )

        manifest = coord.assemble(
            db=db_session,
            contract=contract,
            graph_provider=seeded_graph["engine"],
        )

        assert manifest.graph_enabled is True
        assert manifest.graph_used is True
        assert manifest.graph_entities_count >= 2
        assert manifest.graph_relationships_count >= 1
        assert "## RELATIONAL CONTEXT (KNOWLEDGE GRAPH)" in manifest.system_prompt_snapshot
        assert "Carlos Pérez" in manifest.system_prompt_snapshot
        assert "TechCorp Inc" in manifest.system_prompt_snapshot

    def test_3_graph_provider_failure_failsafe(self, db_session, seeded_graph):
        """
        TEST 3: Fail-Safe.
        If GraphProvider.query() raises an exception, assemble() MUST NOT fail.
        Runtime degrades safely to RAG-only, graph_used=False.
        """
        class FailingGraphProvider(GraphProvider):
            def is_available(self) -> bool:
                return True

            def query(self, graph_query: GraphQuery) -> GraphQueryResult:
                raise RuntimeError("Simulated unexpected graph storage failure!")

        coord = PureCoordinator()
        contract = _make_contract(
            session_id=seeded_graph["session_id"],
            user_message="¿Qué conecta a Carlos Pérez con TechCorp Inc?",
        )

        # Mock RAG service to verify RAG survives graph failure
        class DummyRAGService:
            def build_context(self, **kwargs):
                return "## CONTEXT FROM DOCUMENTS\nDocument content here."

        manifest = coord.assemble(
            db=db_session,
            contract=contract,
            rag_service=DummyRAGService(),
            enable_rag=True,
            graph_provider=FailingGraphProvider(),
        )

        # Assemble must succeed without raising
        assert manifest is not None
        assert manifest.graph_used is False
        # RAG context is intact
        assert "Document content here." in manifest.system_prompt_snapshot
        assert "## RELATIONAL CONTEXT" not in manifest.system_prompt_snapshot

    def test_4_no_project_chat_graceful_degradation(self, db_session, seeded_graph):
        """
        TEST 4: Project Isolation.
        Session without a ProjectChat mapping has project_id=None.
        Graph must NOT execute, graph_used=False, activation reason logged.
        """
        coord = PureCoordinator()
        # Session with no ProjectChat row
        contract = _make_contract(
            session_id="orphan-session-999",
            user_message="¿Cómo se relaciona Carlos Pérez con TechCorp Inc?",
        )

        manifest = coord.assemble(
            db=db_session,
            contract=contract,
            graph_provider=seeded_graph["engine"],
        )

        assert manifest.graph_enabled is True
        assert manifest.graph_used is False
        assert manifest.graph_activation_reason == "No project associated with active session"
        assert "## RELATIONAL CONTEXT" not in manifest.system_prompt_snapshot

    def test_5_graph_provider_none_backward_compatibility(self, db_session, seeded_graph):
        """
        TEST 5: Backward Compatibility.
        Calling assemble() with graph_provider=None must produce identical behavior
        to pre-Gate 7 execution (graph_enabled=False, graph_used=False).
        """
        coord = PureCoordinator()
        contract = _make_contract(
            session_id=seeded_graph["session_id"],
            user_message="¿Cómo se relaciona Carlos Pérez con TechCorp Inc?",
        )

        # 1. Call with default (no graph_provider argument)
        manifest_default = coord.assemble(
            db=db_session,
            contract=contract,
        )

        # 2. Call with explicit None
        manifest_explicit_none = coord.assemble(
            db=db_session,
            contract=contract,
            graph_provider=None,
        )

        assert manifest_default.graph_enabled is False
        assert manifest_default.graph_used is False
        assert manifest_explicit_none.graph_enabled is False
        assert manifest_explicit_none.graph_used is False
        assert manifest_default.system_prompt_snapshot == manifest_explicit_none.system_prompt_snapshot

    def test_6_char_budget_enforcement(self, db_session, seeded_graph):
        """
        TEST 6: Char Budget Invariant.
        Graph context must never cause the final system prompt to exceed char_budget (16,000).
        If existing context nearly exhausts budget, Graph is bounded or omitted.
        """
        coord = PureCoordinator()
        contract = _make_contract(
            session_id=seeded_graph["session_id"],
            user_message="¿Cómo se relaciona Carlos Pérez con TechCorp Inc?",
        )

        # Mock RAG service that returns a huge context (e.g. 15,900 chars)
        class HugeRAGService:
            def build_context(self, **kwargs):
                return "X" * 15850

        manifest = coord.assemble(
            db=db_session,
            contract=contract,
            rag_service=HugeRAGService(),
            enable_rag=True,
            graph_provider=seeded_graph["engine"],
        )

        assert manifest is not None
        assert len(manifest.system_prompt_snapshot) <= manifest.char_budget
        assert manifest.char_count <= manifest.char_budget
        assert manifest.char_budget == 16000

    def test_7_strict_graph_provider_contract_compatibility(self, db_session, seeded_graph):
        """
        TEST 7: Contract Compatibility (Pure Abstract Provider).
        A provider implementing query(graph_query) WITHOUT the 'db' parameter
        must be invoked successfully without TypeError.
        """
        class StrictPureGraphProvider(GraphProvider):
            def is_available(self) -> bool:
                return True

            def query(self, graph_query: GraphQuery) -> GraphQueryResult:
                # Pure provider matching contracts.py exactly (no 'db' param)
                e1 = GraphEntity(id="e1", label="Entidad Alfa", entity_type="concept")
                e2 = GraphEntity(id="e2", label="Entidad Beta", entity_type="concept")
                r1 = GraphRelationship(source_id="e1", target_id="e2", relation_type="conecta_con")
                return GraphQueryResult(
                    entities=[e1, e2],
                    relationships=[r1],
                    graph_available=True,
                )

        coord = PureCoordinator()
        contract = _make_contract(
            session_id=seeded_graph["session_id"],
            user_message="¿Qué relación existe entre Entidad Alfa y Entidad Beta?",
        )

        manifest = coord.assemble(
            db=db_session,
            contract=contract,
            graph_provider=StrictPureGraphProvider(),
        )

        assert manifest.graph_enabled is True
        assert manifest.graph_used is True
        assert manifest.graph_entities_count == 2
        assert manifest.graph_relationships_count == 1
        assert "Entidad Alfa" in manifest.system_prompt_snapshot
        assert "Entidad Beta" in manifest.system_prompt_snapshot

    def test_8_concrete_graph_query_engine_with_db_dispatch(self, db_session, seeded_graph):
        """
        TEST 8: Contract Compatibility (Concrete GraphQueryEngine).
        GraphQueryEngine declaring query(graph_query, db=None) must receive db
        via signature inspection and execute multi-hop query correctly.
        """
        coord = PureCoordinator()
        contract = _make_contract(
            session_id=seeded_graph["session_id"],
            user_message="¿Cómo se vincula Carlos Pérez con TechCorp Inc a través de la cadena de relaciones?",
        )

        manifest = coord.assemble(
            db=db_session,
            contract=contract,
            graph_provider=seeded_graph["engine"],
        )

        assert manifest.graph_enabled is True
        assert manifest.graph_used is True
        assert "Contrato Marco" in manifest.system_prompt_snapshot
