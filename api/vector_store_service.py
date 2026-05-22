"""
AS Code — Vector Store Service (FAISS)

Handles semantic search over chunk embeddings.
SQLite handles chunk *metadata*; this handles chunk *vectors*.

Design:
  - IndexFlatIP: inner product → cosine similarity on L2-normalized vectors
  - int→str ID mapping bridging FAISS positions to chunk UUIDs
  - Disk-persistent: faiss.index + id_mapping.pkl
  - Swappable: Pinecone / Weaviate / Milvus in future phases
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreService:
    """FAISS-backed vector store with disk persistence."""

    def __init__(self, index_path: str, embedding_dim: int = 384):
        import faiss

        self._faiss = faiss
        self.index_path = Path(index_path)
        self.mapping_path = self.index_path.parent / "id_mapping.pkl"
        self.dim = embedding_dim

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or create index
        if self.index_path.exists():
            logger.info(f"Loading FAISS index: {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
        else:
            logger.info(f"Creating FAISS IndexFlatIP (dim={self.dim})")
            self.index = faiss.IndexFlatIP(self.dim)

        # Load or create ID mapping  {faiss_position: chunk_uuid}
        if self.mapping_path.exists():
            with open(self.mapping_path, "rb") as f:
                self._id_map: dict[int, str] = pickle.load(f)
        else:
            self._id_map: dict[int, str] = {}

    # ── Write ──────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
        """Add embeddings. embeddings shape: (N, dim), already normalized."""
        if len(embeddings) != len(chunk_ids):
            raise ValueError("embeddings and chunk_ids length mismatch")

        embeddings = embeddings.astype(np.float32)
        start = self.index.ntotal
        self.index.add(embeddings)

        for i, cid in enumerate(chunk_ids):
            self._id_map[start + i] = cid

        self._persist()
        logger.info(f"Added {len(chunk_ids)} vectors. Index total: {self.index.ntotal}")

    # ── Read ───────────────────────────────────────────────────

    def search(
        self, query: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search top-k nearest vectors.

        Args:
            query: shape (dim,) or (1, dim), L2-normalized float32
            k: number of results

        Returns:
            [(chunk_id, score), ...] sorted by score descending
            Score is inner product ≈ cosine similarity ∈ [0, 1]
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        q = query.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(q, k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            cid = self._id_map.get(int(idx))
            if cid:
                results.append((cid, float(score)))

        return results  # already sorted by FAISS (desc for IP)

    # ── Delete ─────────────────────────────────────────────────

    def remove_chunks(self, chunk_ids: List[str]) -> None:
        """
        Remove vectors by chunk IDs.

        IndexFlatIP has no native remove — we null out the mapping.
        Stale vectors remain but will never be returned (no ID mapping).
        For large-scale cleanup, rebuild the index explicitly.
        """
        ids_to_remove = set(chunk_ids)
        removed = 0
        for pos, cid in list(self._id_map.items()):
            if cid in ids_to_remove:
                del self._id_map[pos]
                removed += 1
        if removed:
            self._persist()
            logger.info(f"Unmapped {removed} vectors (stale slots remain)")

    # ── Persistence ────────────────────────────────────────────

    def _persist(self) -> None:
        self._faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, "wb") as f:
            pickle.dump(self._id_map, f)

    @property
    def total(self) -> int:
        return self.index.ntotal


# ── Singleton ──────────────────────────────────────────────────

_vector_store: Optional[VectorStoreService] = None


def get_vector_store(
    index_path: str = "data/embeddings/faiss.index",
    embedding_dim: int = 384,
) -> VectorStoreService:
    """Return the global VectorStoreService singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService(index_path, embedding_dim)
    return _vector_store
