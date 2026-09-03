"""
AS Code — GATE 6: Graph Trigger & Relational Context Formatter Test Suite

Validates:
  1. Trigger - Negative Cases (RAG Natural Queries -> Graph OFF):
     - "¿Qué dice el contrato?" -> OFF
     - "Resume el documento" -> OFF
     - "Explícame este código" -> OFF
     - "¿Qué personas aparecen en el contrato?" -> OFF
     - Empty query -> OFF
  2. Trigger - Positive Cases (Relational Need -> Graph ON):
     - "¿Qué relación existe entre Juan Pérez y Empresa X?" -> ON
     - "¿Cómo se relaciona el módulo Auth con Database?" -> ON
     - "¿Qué documentos conectan a Juan con Pedro?" -> ON
     - "¿Quién representa a la empresa X en el contrato Y?" -> ON
     - "Módulos que dependen de RuntimeCoordinator" -> ON (programming)
     - "¿Qué contratos están relacionados con el proveedor?" -> ON (legal/business)
     - Multi-hop ("a través de") -> ON
  3. Trigger - Determinism & Explainability:
     - Repeated evaluations return identical decision and signals.
     - Structured signals list is populated.
  4. Formatter - Structure & Content:
     - Formats entities with types.
     - Formats relationships with resolved human-readable entity labels.
     - Formats source document provenance.
  5. Formatter - Edge & Fallback Cases:
     - Empty result -> returns empty string (no hallucination).
     - Graph unavailable -> returns empty string.
     - Bounded character budget enforced.
     - Deterministic ordering of entities and relationships.
"""

import pytest

from runtime.graph.contracts import GraphEntity, GraphQueryResult, GraphRelationship
from runtime.graph.formatter import RelationalContextFormatter
from runtime.graph.trigger import GraphTrigger


# ── 1. Trigger Tests ─────────────────────────────────────────────


class TestGraphTrigger:
    @pytest.fixture
    def trigger(self):
        return GraphTrigger()

    def test_empty_query_is_off(self, trigger):
        dec = trigger.evaluate("")
        assert dec.needed is False
        assert "Empty" in dec.reason

    def test_rag_content_queries_are_off(self, trigger):
        content_queries = [
            "¿Qué dice este contrato sobre las penalidades?",
            "Resume el documento por favor",
            "¿Qué establece la cláusula 5?",
            "¿Cómo funciona esta función?",
            "Explícame este código paso a paso",
            "¿Qué personas aparecen en el contrato?",
            "¿Qué empresas aparecen en estos documentos?",
        ]
        for query in content_queries:
            dec = trigger.evaluate(query)
            assert dec.needed is False, f"Expected OFF for content query: {query}"
            assert "RAG" in dec.reason or "No relational" in dec.reason

    def test_explicit_relational_queries_are_on(self, trigger):
        rel_queries = [
            ("¿Qué relación existe entre Juan Pérez y Empresa X?", "explicit_relation"),
            ("¿Cómo se relaciona el módulo Auth con Database?", "connection_question"),
            ("¿Qué documentos conectan a Juan con Pedro?", "cross_document_connection"),
            ("¿Quién representa a la empresa X en el contrato Y?", "representation_query"),
            ("¿Qué vínculos hay entre ambos socios?", "explicit_relation"),
            ("Conexiones a través de contratos firmados", "multi_hop_chain"),
        ]
        for query, expected_signal in rel_queries:
            dec = trigger.evaluate(query)
            assert dec.needed is True, f"Expected ON for query: {query}"
            assert expected_signal in dec.signals or len(dec.signals) > 0
            assert "Relational query detected" in dec.reason

    def test_domain_programming_dependency(self, trigger):
        dec = trigger.evaluate("¿Qué módulo depende de RuntimeCoordinator?", domain="programming")
        assert dec.needed is True
        assert any("dependency" in s for s in dec.signals)

    def test_domain_legal_cross_contract(self, trigger):
        dec = trigger.evaluate("¿Qué personas firmaron ambos contratos?", domain="legal")
        assert dec.needed is True
        assert "legal_multi_contract" in dec.signals

    def test_trigger_determinism(self, trigger):
        q = "¿Cómo se relaciona el proveedor con el cliente a través de intermediarios?"
        d1 = trigger.evaluate(q)
        d2 = trigger.evaluate(q)
        assert d1.needed == d2.needed
        assert d1.reason == d2.reason
        assert d1.signals == d2.signals


# ── 2. Formatter Tests ───────────────────────────────────────────


class TestRelationalContextFormatter:
    @pytest.fixture
    def formatter(self):
        return RelationalContextFormatter()

    def test_format_empty_or_unavailable(self, formatter):
        assert formatter.format(None) == ""
        assert formatter.format(GraphQueryResult.empty()) == ""
        assert formatter.format(GraphQueryResult.unavailable()) == ""

    def test_format_valid_relational_result(self, formatter):
        e1 = GraphEntity(id="e1", label="Juan Pérez", entity_type="person")
        e2 = GraphEntity(id="e2", label="Contrato A", entity_type="contract")
        e3 = GraphEntity(id="e3", label="Alpha Corp", entity_type="org")

        r1 = GraphRelationship(source_id="e1", target_id="e2", relation_type="firma")
        r2 = GraphRelationship(source_id="e1", target_id="e3", relation_type="asociado_a")

        result = GraphQueryResult(
            entities=[e1, e2, e3],
            relationships=[r1, r2],
            source_document_ids=["doc-1", "doc-2"],
            graph_available=True,
        )

        formatted = formatter.format(result)

        assert "## RELATIONAL CONTEXT (KNOWLEDGE GRAPH)" in formatted
        assert "### Entities:" in formatted
        assert "- Juan Pérez (person)" in formatted
        assert "- Contrato A (contract)" in formatted
        assert "- Alpha Corp (org)" in formatted
        assert "### Relationships:" in formatted
        assert "- Juan Pérez --firma--> Contrato A" in formatted
        assert "- Juan Pérez --asociado_a--> Alpha Corp" in formatted
        assert "### Source Documents:" in formatted
        assert "- doc-1" in formatted
        assert "- doc-2" in formatted

    def test_format_determinism(self, formatter):
        e1 = GraphEntity(id="e1", label="Beta", entity_type="concept")
        e2 = GraphEntity(id="e2", label="Alpha", entity_type="concept")
        r1 = GraphRelationship(source_id="e1", target_id="e2", relation_type="rel_b")
        r2 = GraphRelationship(source_id="e2", target_id="e1", relation_type="rel_a")

        res = GraphQueryResult(entities=[e1, e2], relationships=[r1, r2], graph_available=True)

        txt1 = formatter.format(res)
        txt2 = formatter.format(res)
        assert txt1 == txt2

        # Entities should be sorted alphabetically: Alpha before Beta
        pos_alpha = txt1.find("Alpha")
        pos_beta = txt1.find("Beta")
        assert pos_alpha < pos_beta

    def test_bounded_character_budget(self):
        tiny_formatter = RelationalContextFormatter(max_chars=80)
        e1 = GraphEntity(id="e1", label="Very Long Entity Name A", entity_type="concept")
        e2 = GraphEntity(id="e2", label="Very Long Entity Name B", entity_type="concept")
        r1 = GraphRelationship(source_id="e1", target_id="e2", relation_type="relates_to")

        res = GraphQueryResult(entities=[e1, e2], relationships=[r1], graph_available=True)
        out = tiny_formatter.format(res)
        assert len(out) <= 130 # 80 + suffix length
        assert "[Relational context truncated to fit budget]" in out
