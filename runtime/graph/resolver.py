"""
AS Code — Conservative Entity Resolver

Implements progressive, conservative entity resolution:
  - Strict project isolation: candidates must belong to the same project_id.
  - Layer 1: Exact normalized match (high confidence).
  - Layer 2: Alias / known surface forms match.
  - Layer 3: Normalized key match (stripping accents and legal corporate suffixes).
  - Ambiguity rule: When evidence is insufficient or ambiguous, DO NOT MERGE.
    Preserves false splits over false merges.
  - Persists resolved nodes and edges into GraphStore without direct SQLAlchemy queries.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from api.graph_models import GraphNode
from runtime.graph.extractor import ExtractedEntity, ExtractedRelationship, ExtractionResult
from runtime.graph.normalizer import normalize_key, normalize_label
from runtime.graph.store import GraphStore

logger = logging.getLogger("as-code.runtime.graph.resolver")


class EntityResolver:
    """
    Resolves extracted entities into existing GraphStore nodes within a project.
    Strictly conservative: only merges when there is confident evidence of identity.
    """

    def __init__(self, store: Optional[GraphStore] = None):
        self.store = store or GraphStore()

    def resolve_and_save_node(
        self,
        db: Session,
        project_id: str,
        extracted: ExtractedEntity,
    ) -> Optional[GraphNode]:
        """
        Resolve an ExtractedEntity against existing project nodes and save it via GraphStore.
        
        Resolution Strategy (Project-scoped):
          1. Exact normalized label match.
          2. Matching existing alias.
          3. Suffix / diacritic stripped key match (for compatible entity types only).
          4. Default: Create new entity (No merge if ambiguous).
        """
        norm_lbl = normalize_label(extracted.label)
        if not norm_lbl:
            return None

        # 1. Exact normalized label match
        existing_node = self.store.find_node_by_label(db, project_id, norm_lbl)
        if existing_node:
            # If entity types are incompatible (e.g., person vs contract with identical name),
            # treat as ambiguous and DO NOT merge
            if existing_node.entity_type != extracted.entity_type and existing_node.entity_type != "concept":
                logger.debug(
                    f"[RESOLVER] Type mismatch for label {norm_lbl!r}: "
                    f"existing={existing_node.entity_type} vs new={extracted.entity_type}. Splitting."
                )
                # Save as new distinct node by appending entity_type to prevent unique constraint collision
                disambiguated_label = f"{extracted.label} ({extracted.entity_type})"
                return self.store.save_node(
                    db=db,
                    project_id=project_id,
                    label=disambiguated_label,
                    entity_type=extracted.entity_type,
                    source_doc_id=extracted.document_id,
                    aliases=extracted.aliases,
                    metadata=extracted.metadata,
                )

            # Compatible -> Merge via store (merges source_doc_ids and aliases)
            return self.store.save_node(
                db=db,
                project_id=project_id,
                label=existing_node.label,
                entity_type=existing_node.entity_type,
                source_doc_id=extracted.document_id,
                aliases=extracted.aliases + [extracted.label] if extracted.label != existing_node.label else extracted.aliases,
                metadata=extracted.metadata,
            )

        # 2. Check existing project nodes for alias or stripped key match
        project_nodes = self.store.list_nodes(db, project_id)
        norm_k = normalize_key(extracted.label)

        for pnode in project_nodes:
            # Type compatibility check
            if pnode.entity_type != extracted.entity_type and pnode.entity_type != "concept":
                continue

            # Alias match
            pnode_aliases = [normalize_label(a) for a in pnode.aliases]
            if norm_lbl in pnode_aliases:
                logger.debug(f"[RESOLVER] Alias matched: {extracted.label!r} matches alias of node {pnode.id}")
                return self.store.save_node(
                    db=db,
                    project_id=project_id,
                    label=pnode.label,
                    entity_type=pnode.entity_type,
                    source_doc_id=extracted.document_id,
                    aliases=[extracted.label],
                )

            # Stripped key match (e.g. 'Empresa Alpha S.A.' and 'Empresa Alpha')
            if norm_k and normalize_key(pnode.label) == norm_k:
                logger.debug(f"[RESOLVER] Key matched: {extracted.label!r} matches key of node {pnode.label!r}")
                return self.store.save_node(
                    db=db,
                    project_id=project_id,
                    label=pnode.label,
                    entity_type=pnode.entity_type,
                    source_doc_id=extracted.document_id,
                    aliases=[extracted.label],
                )

        # 3. No confident match -> Save as new node
        return self.store.save_node(
            db=db,
            project_id=project_id,
            label=extracted.label,
            entity_type=extracted.entity_type,
            source_doc_id=extracted.document_id,
            aliases=extracted.aliases,
            metadata=extracted.metadata,
        )

    def ingest_extraction_result(
        self,
        db: Session,
        project_id: str,
        extraction: ExtractionResult,
    ) -> Tuple[int, int]:
        """
        Resolve and persist all entities and relationships in an ExtractionResult.
        Supports cross-document relationship resolution against existing project nodes.
        Returns: (nodes_persisted_count, edges_persisted_count)
        """
        label_to_node_id: dict[str, str] = {}

        # 1. Resolve and persist entities
        nodes_count = 0
        for ent in extraction.entities:
            node = self.resolve_and_save_node(db, project_id, ent)
            if node:
                nodes_count += 1
                # Map both original and normalized labels to this node id for relationship resolution
                label_to_node_id[normalize_label(ent.label)] = node.id
                label_to_node_id[ent.label] = node.id
                for a in ent.aliases:
                    label_to_node_id[normalize_label(a)] = node.id

        # 2. Resolve and persist relationships
        edges_count = 0
        for rel in extraction.relationships:
            src_norm = normalize_label(rel.source_label)
            tgt_norm = normalize_label(rel.target_label)

            src_id = label_to_node_id.get(src_norm) or label_to_node_id.get(rel.source_label)
            tgt_id = label_to_node_id.get(tgt_norm) or label_to_node_id.get(rel.target_label)

            # Cross-document resolution: search existing nodes in the same project
            if not src_id:
                existing_src = self.store.find_node_by_label(db, project_id, src_norm)
                if existing_src:
                    src_id = existing_src.id
                else:
                    for pnode in self.store.list_nodes(db, project_id):
                        if src_norm in [normalize_label(a) for a in pnode.aliases] or normalize_label(pnode.label) == src_norm:
                            src_id = pnode.id
                            break

            if not tgt_id:
                existing_tgt = self.store.find_node_by_label(db, project_id, tgt_norm)
                if existing_tgt:
                    tgt_id = existing_tgt.id
                else:
                    for pnode in self.store.list_nodes(db, project_id):
                        if tgt_norm in [normalize_label(a) for a in pnode.aliases] or normalize_label(pnode.label) == tgt_norm:
                            tgt_id = pnode.id
                            break

            # If still not found, conservatively auto-persist node so edge provenance is preserved
            if not src_id:
                auto_src = self.resolve_and_save_node(
                    db, project_id,
                    ExtractedEntity(label=rel.source_label, entity_type="concept", document_id=rel.document_id)
                )
                if auto_src:
                    src_id = auto_src.id
                    nodes_count += 1
                    label_to_node_id[src_norm] = auto_src.id

            if not tgt_id:
                auto_tgt = self.resolve_and_save_node(
                    db, project_id,
                    ExtractedEntity(label=rel.target_label, entity_type="concept", document_id=rel.document_id)
                )
                if auto_tgt:
                    tgt_id = auto_tgt.id
                    nodes_count += 1
                    label_to_node_id[tgt_norm] = auto_tgt.id

            if src_id and tgt_id and src_id != tgt_id:
                edge = self.store.save_edge(
                    db=db,
                    project_id=project_id,
                    source_node_id=src_id,
                    target_node_id=tgt_id,
                    relation_type=rel.relation_type,
                    source_doc_id=rel.document_id,
                    confidence=rel.confidence,
                    metadata=rel.metadata,
                )
                if edge:
                    edges_count += 1

        # 3. Update project build status
        if nodes_count > 0:
            current_status = self.store.get_build_status(db, project_id)
            new_status = "full" if edges_count > 0 else "entities_only"
            self.store.update_build_status(db, project_id, new_status)

        return nodes_count, edges_count
