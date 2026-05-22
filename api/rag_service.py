"""
AS Code — RAG Service

Orchestrates the full RAG pipeline with two specialized sub-pipelines:

  ChatRAGPipeline  — conversational retrieval, broad semantic search
  CodeRAGPipeline  — symbol-aware retrieval, code-specific context

Both share the same FAISS index and embedder but differ in:
  - Retrieval mode (hybrid vs semantic)
  - Context composition (normal/thinking vs code mode)
  - Top-k defaults

Full pipeline:
    Document ingest:   chunk → embed → store (FAISS + SQLite)
    Query retrieval:   embed query → FAISS search → BM25 merge → group → compose
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from api.context_builder import NotebookContextBuilder, RetrievedChunk, get_context_builder
from api.chunker_service import ChunkerService, get_chunker
from api.embedder_service import EmbedderService, get_embedder
from api.rag_models import RAGDocument, RAGDocumentChunk
from api.vector_store_service import VectorStoreService, get_vector_store

logger = logging.getLogger(__name__)


# ── BM25 Keyword Retrieval ─────────────────────────────────────


class BM25Index:
    """
    Lightweight BM25 index over chunk texts.

    Rebuilt on-the-fly from SQLite — acceptable for Phase 1 scale.
    For larger corpora, persist and incrementally update.
    """

    def __init__(self, chunks: List[RAGDocumentChunk]):
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        corpus = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Returns [(chunk_id, normalized_score), ...] top-k."""
        if not self._bm25 or not self._chunks:
            return []

        tokens = query.lower().split()
        raw_scores = self._bm25.get_scores(tokens)

        # Normalize to [0, 1]
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        norm_scores = raw_scores / max_score

        indexed = sorted(
            enumerate(norm_scores), key=lambda x: x[1], reverse=True
        )[:k]

        return [(self._chunks[i].id, float(s)) for i, s in indexed if s > 0]


# ── Base Pipeline ──────────────────────────────────────────────


class BaseRAGPipeline:
    """Shared ingest + retrieval logic for both pipelines."""

    DEFAULT_TOP_K = 5
    DEFAULT_TOP_K_DEEP = 8
    PIPELINE_NAME = "base"

    def __init__(
        self,
        embedder: EmbedderService,
        vector_store: VectorStoreService,
        chunker: ChunkerService,
        context_builder: NotebookContextBuilder,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker
        self.context_builder = context_builder

    # ── Ingest ──────────────────────────────────────────────────

    def process_document(self, doc: RAGDocument, db: Session) -> int:
        """
        Full ingest pipeline: chunk → embed → store.

        Args:
            doc: RAGDocument already persisted to SQLite.
            db:  Active session for saving chunks.

        Returns:
            Number of chunks created.
        """
        logger.info(f"[{self.PIPELINE_NAME}] Processing: {doc.filename}")

        # 1. Chunk
        raw_chunks = self.chunker.chunk(doc.content, doc.filename, doc.file_type)
        if not raw_chunks:
            logger.warning(f"No chunks produced for {doc.filename}")
            return 0

        # 2. Embed (batch)
        texts = [c.text for c in raw_chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 3. Persist chunks to SQLite (metadata only)
        db_chunks: List[RAGDocumentChunk] = []
        for i, (raw, _emb) in enumerate(zip(raw_chunks, embeddings)):
            db_chunk = RAGDocumentChunk(
                document_id=doc.id,
                section_name=raw.section_name,
                chunk_index=i,
                chunk_type=raw.chunk_type,
                text=raw.text,
                meta_json=json.dumps(raw.metadata),
            )
            db.add(db_chunk)
            db_chunks.append(db_chunk)

        db.commit()

        # Refresh to get auto-generated IDs
        for c in db_chunks:
            db.refresh(c)

        # 4. Store embeddings in FAISS
        chunk_ids = [c.id for c in db_chunks]
        self.vector_store.add(embeddings, chunk_ids)

        logger.info(
            f"[RAG-INGEST] filename={doc.filename!r} | "
            f"chunks={len(db_chunks)} | "
            f"embed_shape={embeddings.shape} | "
            f"pipeline={self.PIPELINE_NAME}"
        )
        logger.info(
            f"[RAG-INDEX] sqlite_chunks={len(db_chunks)} | "
            f"faiss_vectors={self.vector_store.total} | "
            f"bm25_docs={len(db_chunks)}"
        )
        return len(db_chunks)

    # ── Retrieve ────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        db: Session,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        alpha: float = 0.7,
    ) -> List[RetrievedChunk]:
        """
        Hybrid retrieval: semantic (FAISS) + keyword (BM25) → fused scores.

        Args:
            query:  User query string.
            db:     DB session to load chunk metadata.
            top_k:  Number of final chunks to return.
            mode:   "semantic" | "keyword" | "hybrid"
            alpha:  Blend weight — 1.0 = pure semantic, 0.0 = pure keyword.

        Returns:
            List of RetrievedChunk sorted by fused score descending.
        """
        import time
        start_retrieve = time.perf_counter()
        k = top_k or self.DEFAULT_TOP_K

        # ─ Semantic retrieval ─
        semantic_hits: dict[str, float] = {}
        embedding_time_ms = 0.0
        if mode in ("semantic", "hybrid"):
            start_embed = time.perf_counter()
            q_emb = self.embedder.embed_query(query)
            embedding_time_ms = (time.perf_counter() - start_embed) * 1000
            sem_results = self.vector_store.search(q_emb, k=k * 2)
            semantic_hits = {cid: score for cid, score in sem_results}

        # ─ Keyword retrieval ─
        keyword_hits: dict[str, float] = {}
        if mode in ("keyword", "hybrid"):
            all_chunks = db.query(RAGDocumentChunk).all()
            bm25 = BM25Index(all_chunks)
            kw_results = bm25.search(query, k=k * 2)
            keyword_hits = {cid: score for cid, score in kw_results}

        # ─ Fuse scores ─
        all_ids = set(semantic_hits) | set(keyword_hits)
        fused: dict[str, float] = {}
        for cid in all_ids:
            sem = semantic_hits.get(cid, 0.0)
            kw = keyword_hits.get(cid, 0.0)
            fused[cid] = alpha * sem + (1 - alpha) * kw

        # Top-k by fused score
        top_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:k]

        retrieval_time_ms = (time.perf_counter() - start_retrieve) * 1000

        logger.info(
            f"[RAG-RETRIEVE] query={query!r} | pipeline={self.PIPELINE_NAME} | mode={mode} | "
            f"retrieval_mode={mode} | semantic_hits={len(semantic_hits)} | "
            f"keyword_hits={len(keyword_hits)} | fused_total={len(fused)} | "
            f"top_k={len(top_ids)} | embedding_time_ms={embedding_time_ms:.2f} | "
            f"retrieval_time_ms={retrieval_time_ms:.2f}"
        )

        if not top_ids:
            return []

        # ─ Load chunk metadata from SQLite ─
        db_chunks = (
            db.query(RAGDocumentChunk)
            .filter(RAGDocumentChunk.id.in_(top_ids))
            .all()
        )
        chunk_map = {c.id: c for c in db_chunks}

        results: List[RetrievedChunk] = []
        for cid in top_ids:
            chunk = chunk_map.get(cid)
            if not chunk:
                continue
            doc = chunk.document
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=chunk.text,
                    section_name=chunk.section_name or "Unknown",
                    chunk_type=chunk.chunk_type or "text",
                    document_filename=doc.filename if doc else "unknown",
                    document_id=chunk.document_id,
                    score=fused[cid],
                    metadata=chunk.meta,
                )
            )

        top_chunks_log = [
            {"doc": c.document_filename, "section": c.section_name, "score": round(c.score, 4)}
            for c in results
        ]
        logger.info(f"[RAG-RETRIEVE-CHUNKS] top_chunks={top_chunks_log}")

        return results

    # ── Context ─────────────────────────────────────────────────

    def build_context(
        self,
        query: str,
        db: Session,
        mode: str = "normal",
        top_k: Optional[int] = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.7,
    ) -> str:
        """
        Full pipeline: retrieve → group → compose.

        Returns context string ready for prompt injection.
        """
        import time
        deep = mode == "thinking"
        k = top_k or (self.DEFAULT_TOP_K_DEEP if deep else self.DEFAULT_TOP_K)

        chunks = self.retrieve(query, db, top_k=k, mode=retrieval_mode, alpha=alpha)
        if not chunks:
            return ""

        start_compose = time.perf_counter()
        context = self.context_builder.build(chunks, mode=mode, query=query)
        composition_time_ms = (time.perf_counter() - start_compose) * 1000

        logger.info(
            f"[RAG-COMPOSE] composition_time_ms={composition_time_ms:.2f} | "
            f"context_len={len(context)} | mode={mode}"
        )
        return context


# ── Chat Pipeline ──────────────────────────────────────────────


class ChatRAGPipeline(BaseRAGPipeline):
    """
    Conversational RAG pipeline.

    - Broad hybrid retrieval (semantic + keyword)
    - Normal / thinking context modes
    - Targets: txt, md, pdf, docx documents
    """

    DEFAULT_TOP_K = 5
    DEFAULT_TOP_K_DEEP = 8
    PIPELINE_NAME = "chat"


# ── Code Pipeline ──────────────────────────────────────────────


class CodeRAGPipeline(BaseRAGPipeline):
    """
    Code-aware RAG pipeline.

    - Primarily semantic retrieval (symbol names are exact — BM25 helps too)
    - Code context composition mode
    - Targets: .py, .js, .ts, .go, … source files
    """

    DEFAULT_TOP_K = 6
    DEFAULT_TOP_K_DEEP = 10
    PIPELINE_NAME = "code"

    def build_context(
        self,
        query: str,
        db: Session,
        mode: str = "code",      # default to "code" mode
        top_k: Optional[int] = None,
        retrieval_mode: str = "hybrid",
        alpha: float = 0.65,     # slightly more keyword weight for symbol names
    ) -> str:
        return super().build_context(
            query=query,
            db=db,
            mode=mode,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
        )


# ── Orchestrator ───────────────────────────────────────────────


class RAGService:
    """
    Top-level RAG orchestrator.

    Routes documents and queries to the appropriate pipeline based on
    document type / explicit pipeline parameter.
    """

    def __init__(
        self,
        chat_pipeline: ChatRAGPipeline,
        code_pipeline: CodeRAGPipeline,
        retrieval_mode: str = "hybrid",
        hybrid_alpha: float = 0.7,
    ):
        self.chat = chat_pipeline
        self.code = code_pipeline
        self.retrieval_mode = retrieval_mode
        self.hybrid_alpha = hybrid_alpha

    def _select_pipeline(self, pipeline: str) -> BaseRAGPipeline:
        return self.code if pipeline == "code" else self.chat

    def process_document(self, doc: RAGDocument, db: Session) -> int:
        """Ingest a document through the appropriate pipeline."""
        pipeline = self._select_pipeline(doc.pipeline or "chat")
        return pipeline.process_document(doc, db)

    def build_context(
        self,
        query: str,
        db: Session,
        mode: str = "normal",
        pipeline: str = "chat",
        top_k: Optional[int] = None,
    ) -> str:
        """Build NotebookLM-style context for a query."""
        p = self._select_pipeline(pipeline)
        return p.build_context(
            query=query,
            db=db,
            mode=mode,
            top_k=top_k,
            retrieval_mode=self.retrieval_mode,
            alpha=self.hybrid_alpha,
        )

    def retrieve(
        self,
        query: str,
        db: Session,
        top_k: int = 5,
        pipeline: str = "chat",
    ) -> List[RetrievedChunk]:
        """Raw retrieval without context composition (for testing/debug)."""
        p = self._select_pipeline(pipeline)
        return p.retrieve(
            query=query,
            db=db,
            top_k=top_k,
            mode=self.retrieval_mode,
            alpha=self.hybrid_alpha,
        )


# ── Factory ────────────────────────────────────────────────────


def build_rag_service(
    faiss_index_path: str = "data/embeddings/faiss.index",
    embedding_dim: int = 384,
    embedder_model: str = "BAAI/bge-small-en-v1.5",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    retrieval_mode: str = "hybrid",
    hybrid_alpha: float = 0.7,
) -> RAGService:
    """Construct and return a fully wired RAGService."""
    embedder = get_embedder(embedder_model)
    vector_store = get_vector_store(faiss_index_path, embedding_dim)
    chunker = get_chunker(chunk_size, chunk_overlap)
    context_builder = get_context_builder()

    shared_kwargs = dict(
        embedder=embedder,
        vector_store=vector_store,
        chunker=chunker,
        context_builder=context_builder,
    )

    return RAGService(
        chat_pipeline=ChatRAGPipeline(**shared_kwargs),
        code_pipeline=CodeRAGPipeline(**shared_kwargs),
        retrieval_mode=retrieval_mode,
        hybrid_alpha=hybrid_alpha,
    )
