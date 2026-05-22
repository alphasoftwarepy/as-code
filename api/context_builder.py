"""
AS Code — NotebookLM Context Builder

Implements the core NotebookLM-style context composition pipeline:

    retrieve → group → compose → inject

NOT "top-k chunks → prompt".

Groups retrieved chunks by document × section to preserve information
hierarchy, then renders a structured context block tuned to the
inference mode (normal | thinking | code).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with its relevance score."""

    chunk_id: str
    text: str
    section_name: str
    chunk_type: str          # text | code | markdown_section
    document_filename: str
    document_id: str
    score: float             # cosine similarity ∈ [0, 1]
    metadata: dict = field(default_factory=dict)

    @property
    def relevance_pct(self) -> int:
        return int(min(self.score, 1.0) * 100)

    @property
    def symbol(self) -> Optional[str]:
        return self.metadata.get("symbol")

    @property
    def language(self) -> Optional[str]:
        return self.metadata.get("language")


@dataclass
class ContextGroup:
    """Chunks grouped by (document, section) for hierarchy-aware composition."""

    document: str
    section: str
    chunk_type: str
    chunks: List[RetrievedChunk] = field(default_factory=list)

    @property
    def max_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)

    @property
    def avg_score(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.score for c in self.chunks) / len(self.chunks)


# ── Builder ────────────────────────────────────────────────────


class NotebookContextBuilder:
    """
    Hierarchy-aware context builder.

    Pipeline:
        1. group()   — cluster chunks by (doc, section)
        2. sort()    — rank groups by max relevance score
        3. compose() — render structured string per mode
    """

    def build(
        self,
        chunks: List[RetrievedChunk],
        mode: str = "normal",
        query: Optional[str] = None,
        max_chars: int = 6000,
    ) -> str:
        """
        Main entry point.

        Args:
            chunks: Retrieved + scored chunks (any order).
            mode:   "normal" | "thinking" | "code"
            query:  Original query (used in thinking mode header).
            max_chars: Maximum character budget.

        Returns:
            Formatted context string ready for prompt injection.
        """
        if not chunks:
            return ""

        groups = self._group(chunks)

        if mode == "thinking":
            return self._compose_thinking(groups, query, max_chars=max_chars)
        elif mode == "code":
            return self._compose_code(groups, query, max_chars=max_chars)
        else:
            return self._compose_normal(groups, max_chars=max_chars)

    # ── Step 1: Group ──────────────────────────────────────────

    def _group(self, chunks: List[RetrievedChunk]) -> List[ContextGroup]:
        """
        Cluster chunks by (document_filename, section_name).
        Groups sorted by max relevance score descending.
        """
        bucket: Dict[Tuple[str, str], ContextGroup] = {}

        for chunk in chunks:
            key = (chunk.document_filename, chunk.section_name)
            if key not in bucket:
                bucket[key] = ContextGroup(
                    document=chunk.document_filename,
                    section=chunk.section_name,
                    chunk_type=chunk.chunk_type,
                )
            bucket[key].chunks.append(chunk)

        groups = list(bucket.values())
        groups.sort(key=lambda g: g.max_score, reverse=True)
        return groups

    # ── Step 2 / 3: Compose ────────────────────────────────────

    def _compose_normal(self, groups: List[ContextGroup], max_chars: int = 6000) -> str:
        """
        Normal mode: compact, per-group blocks.

        Format:
            ## CONTEXT FROM DOCUMENTS

            [doc.py › Section] relevance: 87%
            <chunk text>

            [doc2.md › Section] relevance: 72%
            <chunk text>
        """
        lines = ["## CONTEXT FROM DOCUMENTS\n"]
        current_len = sum(len(line) + 1 for line in lines)

        for group in groups:
            pct = int(group.max_score * 100)
            header = f"[{group.document} › {group.section}] relevance: {pct}%"
            if group.chunk_type == "code":
                lang = group.chunks[0].language or "code"
                header += f" [{lang}]"

            # Concatenate all chunks in the group (already ordered by retrieval)
            combined = "\n\n".join(c.text for c in group.chunks)
            # Truncate per-group to avoid context bloat
            if len(combined) > 600:
                combined = combined[:600] + "…"

            if group.chunk_type == "code":
                lang = group.chunks[0].language or ""
                block_content = f"```{lang}\n{combined}\n```"
            else:
                block_content = combined

            block_text = f"{header}\n{block_content}\n\n"
            if current_len + len(block_text) > max_chars:
                lines.append("\n[Context truncated due to size limits]")
                break

            lines.append(header)
            lines.append(block_content)
            lines.append("")
            current_len += len(block_text)

        return "\n".join(lines)

    def _compose_thinking(
        self, groups: List[ContextGroup], query: Optional[str], max_chars: int = 6000
    ) -> str:
        """
        Thinking / deep-research mode: full hierarchy, analysis guidance.

        Format:
            ## RESEARCH CONTEXT — "<query>"

            ### [Document: doc.py]
            #### Section: FunctionName (relevance 91% | python | symbol: register_model)
            <full chunk text>

            ## ANALYSIS GUIDANCE
            1. …
        """
        q_label = f'"{query}"' if query else "query"
        lines = [f"## RESEARCH CONTEXT — {q_label}\n"]
        current_len = sum(len(line) + 1 for line in lines)

        # Guidance block at the end (keep track of its length)
        guidance = [
            "\n## ANALYSIS GUIDANCE",
            "1. Compare retrieved patterns against industry best practices",
            "2. Identify inconsistencies, gaps, or anti-patterns",
            "3. Cross-reference symbols across multiple documents when present",
            "4. Suggest concrete, actionable improvements",
        ]
        guidance_text = "\n".join(guidance)
        guidance_len = len(guidance_text) + 2

        # Group by document first
        by_doc: Dict[str, List[ContextGroup]] = defaultdict(list)
        for g in groups:
            by_doc[g.document].append(g)

        truncated = False
        for doc, doc_groups in by_doc.items():
            if truncated:
                break
            
            doc_header = f"### [Document: {doc}]"
            if current_len + len(doc_header) + guidance_len > max_chars:
                truncated = True
                break
            
            lines.append(doc_header)
            current_len += len(doc_header) + 1

            for group in doc_groups:
                pct = int(group.max_score * 100)
                meta_parts = [f"relevance {pct}%"]
                if group.chunk_type == "code":
                    meta_parts.append(group.chunks[0].language or "code")
                symbols = [c.symbol for c in group.chunks if c.symbol]
                if symbols:
                    meta_parts.append(f"symbols: {', '.join(symbols)}")

                group_header = f"#### {group.section} ({' | '.join(meta_parts)})"
                
                # Render chunks
                group_lines = []
                for chunk in group.chunks:
                    if chunk.chunk_type == "code":
                        lang = chunk.language or ""
                        group_lines.append(f"```{lang}\n{chunk.text}\n```")
                    else:
                        group_lines.append(chunk.text)
                
                group_content = "\n".join(group_lines) + "\n"
                full_group_block = f"{group_header}\n{group_content}\n"
                
                if current_len + len(full_group_block) + guidance_len > max_chars:
                    truncated = True
                    break
                
                lines.append(group_header)
                lines.append(group_content)
                current_len += len(full_group_block)

        if truncated:
            lines.append("\n[Context truncated due to size limits]")

        lines.extend(guidance)
        return "\n".join(lines)

    def _compose_code(
        self, groups: List[ContextGroup], query: Optional[str], max_chars: int = 6000
    ) -> str:
        """
        Code pipeline mode: symbol-first, minimal prose, machine-readable.

        Format:
            ## CODE CONTEXT

            ### symbol: register_model  [engine.py:42-78] relevance: 94%
            ```python
            def register_model(…):
                …
            ```
        """
        lines = ["## CODE CONTEXT\n"]
        current_len = sum(len(line) + 1 for line in lines)

        for group in groups:
            truncated = False
            for chunk in group.chunks:
                pct = chunk.relevance_pct
                sym = chunk.symbol or group.section
                meta = chunk.metadata

                loc = ""
                if meta.get("line_start") and meta.get("line_end"):
                    loc = f"  [{group.document}:{meta['line_start']}-{meta['line_end']}]"
                elif meta.get("file"):
                    loc = f"  [{meta['file']}]"

                lang = chunk.language or ""
                header = f"### symbol: {sym}{loc}  relevance: {pct}%"
                content = f"```{lang}\n{chunk.text}\n```"
                block = f"{header}\n{content}\n\n"

                if current_len + len(block) > max_chars:
                    lines.append("\n[Context truncated due to size limits]")
                    truncated = True
                    break

                lines.append(header)
                lines.append(content)
                lines.append("")
                current_len += len(block)
            
            if truncated:
                break

        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────

_builder: Optional[NotebookContextBuilder] = None


def get_context_builder() -> NotebookContextBuilder:
    """Return the global NotebookContextBuilder singleton."""
    global _builder
    if _builder is None:
        _builder = NotebookContextBuilder()
    return _builder
