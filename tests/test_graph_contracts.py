"""
AS Core — GATE 2: Graph Contracts Test Suite

Validates that:
  - All Graph contracts are correctly defined (GraphEntity, GraphRelationship,
    GraphQuery, GraphQueryResult, GraphProvider).
  - Serialization to/from JSON is clean and self-contained.
  - ContextManifest is fully backward-compatible with the new Graph fields.
  - AS Core functions identically when Graph is OFF (all defaults).
  - GraphProvider is abstract and cannot be instantiated directly.

Does NOT test:
  - SQLite storage (Gate 3)
  - Entity extraction (Gate 4)
  - Graph traversal (Gate 5)
  - Trigger activation (Gate 6)
  - Runtime integration (Gate 7)
  - E2E flows (Gate 10)
"""

import json
import pytest
from abc import ABC
from runtime.graph.contracts import (
    GraphEntity,
    GraphRelationship,
    GraphQuery,
    GraphQueryResult,
    GraphProvider,
)
from runtime.coordinator.models import (
    ContextManifest,
    WorkflowState,
)


# ── GraphEntity ─────────────────────────────────────────────────


class TestGraphEntity:
    def test_minimal_creation(self):
        entity = GraphEntity(id="e1", label="Juan Pérez", entity_type="person")
        assert entity.id == "e1"
        assert entity.label == "Juan Pérez"
        assert entity.entity_type == "person"

    def test_defaults_are_empty_collections(self):
        entity = GraphEntity(id="e2", label="Empresa X", entity_type="org")
        assert entity.source_document_ids == []
        assert entity.aliases == []
        assert entity.metadata == {}

    def test_full_construction(self):
        entity = GraphEntity(
            id="e3",
            label="Empresa X S.A.",
            entity_type="org",
            source_document_ids=["doc-1", "doc-2"],
            aliases=["Empresa X", "EMPRESA X SA"],
            metadata={"domain": "legal"},
        )
        assert len(entity.source_document_ids) == 2
        assert len(entity.aliases) == 2
        assert entity.metadata["domain"] == "legal"

    def test_json_serialization(self):
        entity = GraphEntity(
            id="e4",
            label="Módulo auth",
            entity_type="module",
            source_document_ids=["doc-a"],
            aliases=["auth.py"],
        )
        data = json.loads(entity.model_dump_json())
        assert data["id"] == "e4"
        assert data["label"] == "Módulo auth"
        assert data["entity_type"] == "module"
        assert "doc-a" in data["source_document_ids"]

    def test_json_roundtrip(self):
        entity = GraphEntity(
            id="e5", label="Contrato N°123", entity_type="contract",
            metadata={"date": "2026-01-15"},
        )
        raw = entity.model_dump_json()
        restored = GraphEntity.model_validate_json(raw)
        assert restored.id == entity.id
        assert restored.label == entity.label
        assert restored.metadata["date"] == "2026-01-15"


# ── GraphRelationship ───────────────────────────────────────────


class TestGraphRelationship:
    def test_minimal_creation(self):
        rel = GraphRelationship(
            source_id="e1", target_id="e2", relation_type="firma"
        )
        assert rel.source_id == "e1"
        assert rel.target_id == "e2"
        assert rel.relation_type == "firma"

    def test_defaults(self):
        rel = GraphRelationship(
            source_id="e1", target_id="e2", relation_type="aparece_con"
        )
        assert rel.source_document_id is None
        assert rel.confidence == 1.0
        assert rel.metadata == {}

    def test_full_construction(self):
        rel = GraphRelationship(
            source_id="e1",
            target_id="e2",
            relation_type="propietario",
            source_document_id="doc-5",
            confidence=0.9,
            metadata={"context": "escritura pública"},
        )
        assert rel.source_document_id == "doc-5"
        assert rel.confidence == 0.9
        assert rel.metadata["context"] == "escritura pública"

    def test_json_serialization(self):
        rel = GraphRelationship(
            source_id="e10", target_id="e11", relation_type="importa",
            confidence=0.75,
        )
        data = json.loads(rel.model_dump_json())
        assert data["source_id"] == "e10"
        assert data["target_id"] == "e11"
        assert data["confidence"] == 0.75

    def test_generic_relation_types_for_multiple_domains(self):
        """Relation types are plain strings — no domain subclasses needed."""
        domains = [
            ("e1", "e2", "firma"),           # legal
            ("e3", "e4", "importa"),          # programming
            ("e5", "e6", "cliente_de"),       # business
            ("e7", "e8", "propietario"),      # notarial
            ("e9", "e10", "menciona"),        # generic
        ]
        for src, tgt, rt in domains:
            rel = GraphRelationship(source_id=src, target_id=tgt, relation_type=rt)
            assert rel.relation_type == rt


# ── GraphQuery ──────────────────────────────────────────────────


class TestGraphQuery:
    def test_minimal_creation_requires_project_id_and_query(self):
        q = GraphQuery(project_id="proj-abc", query="¿Quién firma el contrato?")
        assert q.project_id == "proj-abc"
        assert q.query == "¿Quién firma el contrato?"

    def test_defaults(self):
        q = GraphQuery(project_id="proj-xyz", query="test")
        assert q.domain is None
        assert q.max_depth == 2
        assert q.max_nodes == 30
        assert q.timeout_seconds == 5.0

    def test_project_id_mandatory(self):
        with pytest.raises(Exception):
            GraphQuery(query="missing project")

    def test_query_mandatory(self):
        with pytest.raises(Exception):
            GraphQuery(project_id="proj-1")

    def test_custom_bounds(self):
        q = GraphQuery(
            project_id="proj-1",
            query="multi-hop",
            max_depth=3,
            max_nodes=50,
            timeout_seconds=8.0,
        )
        assert q.max_depth == 3
        assert q.max_nodes == 50
        assert q.timeout_seconds == 8.0

    def test_domain_hint(self):
        q = GraphQuery(project_id="p", query="q", domain="legal")
        assert q.domain == "legal"

    def test_session_id_is_not_a_field(self):
        """session_id must NOT be part of GraphQuery — isolation is by project_id."""
        q = GraphQuery(project_id="p", query="q")
        assert not hasattr(q, "session_id")

    def test_json_serialization(self):
        q = GraphQuery(project_id="proj-99", query="¿dependencias?", domain="programming")
        data = json.loads(q.model_dump_json())
        assert data["project_id"] == "proj-99"
        assert data["domain"] == "programming"


# ── GraphQueryResult ────────────────────────────────────────────


class TestGraphQueryResult:
    def test_empty_factory(self):
        result = GraphQueryResult.empty()
        assert result.graph_available is True
        assert result.entities == []
        assert result.relationships == []
        assert result.source_document_ids == []

    def test_unavailable_factory(self):
        result = GraphQueryResult.unavailable()
        assert result.graph_available is False
        assert result.entities == []
        assert result.relationships == []

    def test_full_result(self):
        entity = GraphEntity(id="e1", label="Juan", entity_type="person")
        rel = GraphRelationship(source_id="e1", target_id="e2", relation_type="firma")
        result = GraphQueryResult(
            entities=[entity],
            relationships=[rel],
            source_document_ids=["doc-1"],
            graph_available=True,
            metadata={"traversal_depth": "2"},
        )
        assert len(result.entities) == 1
        assert len(result.relationships) == 1
        assert "doc-1" in result.source_document_ids

    def test_json_serialization_no_internal_deps(self):
        """GraphQueryResult must serialize to plain JSON with no runtime objects."""
        entity = GraphEntity(id="e1", label="Test", entity_type="concept")
        result = GraphQueryResult(
            entities=[entity],
            relationships=[],
            source_document_ids=["doc-x"],
            graph_available=True,
        )
        raw = result.model_dump_json()
        data = json.loads(raw)
        assert data["graph_available"] is True
        assert data["entities"][0]["label"] == "Test"
        assert isinstance(data["relationships"], list)

    def test_json_roundtrip(self):
        result = GraphQueryResult(
            entities=[GraphEntity(id="e99", label="Módulo X", entity_type="module")],
            relationships=[],
            source_document_ids=["doc-z"],
            graph_available=True,
            metadata={"note": "partial"},
        )
        raw = result.model_dump_json()
        restored = GraphQueryResult.model_validate_json(raw)
        assert restored.entities[0].id == "e99"
        assert restored.metadata["note"] == "partial"
        assert restored.graph_available is True

    def test_unavailable_result_is_json_serializable(self):
        result = GraphQueryResult.unavailable()
        raw = result.model_dump_json()
        data = json.loads(raw)
        assert data["graph_available"] is False


# ── GraphProvider ───────────────────────────────────────────────


class TestGraphProvider:
    def test_cannot_instantiate_abstract_provider(self):
        """GraphProvider is abstract — direct instantiation must fail."""
        with pytest.raises(TypeError):
            GraphProvider()

    def test_concrete_subclass_must_implement_both_methods(self):
        """A partial implementation raises TypeError on instantiation."""
        class IncompleteProvider(GraphProvider):
            def is_available(self) -> bool:
                return False
            # Missing query() method

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_minimal_concrete_implementation(self):
        class NoOpProvider(GraphProvider):
            def is_available(self) -> bool:
                return False

            def query(self, graph_query: GraphQuery) -> GraphQueryResult:
                return GraphQueryResult.unavailable()

        p = NoOpProvider()
        assert p.is_available() is False
        result = p.query(GraphQuery(project_id="p", query="q"))
        assert result.graph_available is False

    def test_provider_returns_empty_when_available_no_matches(self):
        class EmptyProvider(GraphProvider):
            def is_available(self) -> bool:
                return True

            def query(self, graph_query: GraphQuery) -> GraphQueryResult:
                return GraphQueryResult.empty()

        p = EmptyProvider()
        result = p.query(GraphQuery(project_id="p", query="test"))
        assert result.graph_available is True
        assert result.entities == []

    def test_provider_interface_is_abc(self):
        assert issubclass(GraphProvider, ABC)


# ── ContextManifest backward compatibility ──────────────────────


class TestContextManifestBackwardCompatibility:
    """
    Verify that the existing ContextManifest construction sites work unchanged
    and that the new Graph fields default correctly to the OFF state.
    """

    def _make_manifest(self, **overrides) -> ContextManifest:
        """Minimal valid manifest matching the existing construction in routes.py."""
        base = dict(
            contract_id="req-test",
            active_skill=None,
            workflow_state=WorkflowState(),
            rag_enabled=False,
            system_prompt_snapshot="[LANG=ES]\nTest prompt",
        )
        base.update(overrides)
        return ContextManifest(**base)

    def test_existing_construction_unchanged(self):
        """ContextManifest can be built exactly as before without graph fields."""
        manifest = self._make_manifest()
        assert manifest.contract_id == "req-test"
        assert manifest.rag_enabled is False

    def test_graph_fields_default_to_off(self):
        manifest = self._make_manifest()
        assert manifest.graph_enabled is False
        assert manifest.graph_used is False
        assert manifest.graph_entities_count == 0
        assert manifest.graph_relationships_count == 0
        assert manifest.graph_activation_reason is None

    def test_graph_off_serialization_unchanged(self):
        """Serialization with graph OFF must not pollute existing fields."""
        manifest = self._make_manifest()
        data = json.loads(manifest.model_dump_json(exclude={"system_prompt_snapshot"}))
        # Existing fields still present
        assert "contract_id" in data
        assert "rag_enabled" in data
        # Graph fields present with safe defaults
        assert data["graph_enabled"] is False
        assert data["graph_used"] is False
        assert data["graph_entities_count"] == 0

    def test_graph_on_state_serializable(self):
        """When Graph activates (future gate), manifest remains serializable."""
        manifest = self._make_manifest(
            graph_enabled=True,
            graph_used=True,
            graph_entities_count=12,
            graph_relationships_count=8,
            graph_activation_reason="multi_document_relational_query",
        )
        data = json.loads(manifest.model_dump_json(exclude={"system_prompt_snapshot"}))
        assert data["graph_enabled"] is True
        assert data["graph_used"] is True
        assert data["graph_entities_count"] == 12
        assert data["graph_relationships_count"] == 8
        assert data["graph_activation_reason"] == "multi_document_relational_query"

    def test_model_dump_json_exclude_still_works(self):
        """Existing exclude pattern used in routes.py logger must continue working."""
        manifest = self._make_manifest()
        dumped = manifest.model_dump_json(exclude={"system_prompt_snapshot"})
        data = json.loads(dumped)
        assert "system_prompt_snapshot" not in data
        assert "contract_id" in data

    def test_as_core_works_with_graph_always_off(self):
        """
        Simulate the full current ContextManifest usage from routes.py fallback:
        graph is never set, everything defaults. This is the 'Graph never implemented'
        scenario — AS Core must function identically.
        """
        from runtime.coordinator.models import WorkflowState
        manifest = ContextManifest(
            contract_id="req-fallback",
            active_skill=None,
            workflow_state=WorkflowState(),
            rag_enabled=False,
            rag_query=None,
            char_budget=16000,
            char_count=250,
            system_prompt_snapshot="[LANG=ES]\nGeneral prompt",
            capability_gate_open=False,
        )
        # No graph fields set at all — must work and default to OFF
        assert manifest.graph_enabled is False
        assert manifest.graph_used is False
        assert manifest.graph_entities_count == 0
        assert manifest.graph_relationships_count == 0
        assert manifest.graph_activation_reason is None
