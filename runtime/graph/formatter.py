"""
AS Code — Relational Context Formatter

Transforms structured GraphQueryResult into a compact, bounded, and deterministic
text block suitable for downstream context assembly.

Principles:
  - Structure only: does NOT generate answers, explanations, or prose.
  - Zero LLM calls: pure deterministic formatting.
  - Strict preservation of entity types, relationship verbs, and source document provenance.
  - Bounded size: respects character budgets deterministically without breaking relationships.
"""

from __future__ import annotations

from typing import Optional

from runtime.graph.contracts import GraphQueryResult


class RelationalContextFormatter:
    """
    Renders structured GraphQueryResult into clean, readable relational context blocks.
    """

    def __init__(self, max_chars: int = 4000):
        self.max_chars = max_chars

    def format(self, result: Optional[GraphQueryResult]) -> str:
        """
        Format a GraphQueryResult into a structured relational context block.
        
        Rules:
          - If result is None or not available -> returns empty string (graceful fallback).
          - If result is empty (no entities/relationships) -> returns empty string.
          - Otherwise formats:
              ## RELATIONAL CONTEXT (KNOWLEDGE GRAPH)
              ### Entities:
              - Label (type)
              ### Relationships:
              - Source --relation--> Target
              ### Sources:
              - doc-id
        """
        if not result or not result.graph_available:
            return ""

        if not result.entities and not result.relationships:
            return ""

        lines: list[str] = ["## RELATIONAL CONTEXT (KNOWLEDGE GRAPH)"]

        # 1. Format Entities (sorted deterministically)
        if result.entities:
            lines.append("### Entities:")
            sorted_entities = sorted(result.entities, key=lambda e: (e.label.lower(), e.id))
            for entity in sorted_entities:
                type_str = f" ({entity.entity_type})" if entity.entity_type else ""
                lines.append(f"- {entity.label}{type_str}")

        # 2. Format Relationships (with resolved entity labels for clear readability)
        if result.relationships:
            lines.append("### Relationships:")
            # Build label lookup by id
            id_to_label = {e.id: e.label for e in result.entities}
            
            sorted_rels = sorted(
                result.relationships,
                key=lambda r: (r.relation_type, id_to_label.get(r.source_id, r.source_id), id_to_label.get(r.target_id, r.target_id))
            )
            
            for rel in sorted_rels:
                src_label = id_to_label.get(rel.source_id, rel.source_id)
                tgt_label = id_to_label.get(rel.target_id, rel.target_id)
                lines.append(f"- {src_label} --{rel.relation_type}--> {tgt_label}")

        # 3. Format Provenance / Sources
        if result.source_document_ids:
            lines.append("### Source Documents:")
            for doc_id in sorted(result.source_document_ids):
                lines.append(f"- {doc_id}")

        text = "\n".join(lines)

        # 4. Enforce Character Budget
        if len(text) > self.max_chars:
            # Bounded truncation without crashing
            text = text[:self.max_chars].rstrip() + "\n[Relational context truncated to fit budget]"

        return text
