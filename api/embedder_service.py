"""
AS Code — Embedder Service

Uses BAAI/bge-small-en-v1.5 (384-dim, local).

Chosen over all-MiniLM-L6-v2 because:
  - Better semantic ranking for technical/code content
  - Instruction-aware query embedding (BGE prefix)
  - Same size footprint (~22MB), fully offline
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBEDDING_DIM = 384

# Query instruction used by BGE models for asymmetric retrieval
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbedderService:
    """
    Embedding service — BAAI/bge-small-en-v1.5.

    Swappable: replace with OpenAI, Claude, or Cohere embeddings
    by subclassing and updating the singleton below.
    """

    def __init__(self, model_name: str = _MODEL_NAME):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedder: {model_name}")
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = _EMBEDDING_DIM
        logger.info(f"Embedder ready — dim={self.dim}")

    # ── Public API ─────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of document chunks.
        Returns shape (N, 384) float32, L2-normalized.
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a retrieval query with BGE instruction prefix.
        Returns shape (384,) float32, L2-normalized.
        """
        instructed = _QUERY_INSTRUCTION + query
        embedding = self._model.encode(
            instructed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text without query instruction. Shape (384,)."""
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)


# ── Singleton ──────────────────────────────────────────────────

_embedder: Optional[EmbedderService] = None


def get_embedder(model_name: str = _MODEL_NAME) -> EmbedderService:
    """Return the global EmbedderService singleton."""
    global _embedder
    if _embedder is None:
        _embedder = EmbedderService(model_name)
    return _embedder
