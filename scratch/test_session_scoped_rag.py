import sys
import os
from unittest.mock import MagicMock
import numpy as np

sys.path.append("c:/as-code")

# Mock embedding/vector store services before importing other modules
import api.embedder_service
import api.vector_store_service
api.embedder_service.get_embedder = MagicMock()
api.vector_store_service.get_vector_store = MagicMock()

from api.database import init_db, get_session
from api.rag_models import RAGDocument, RAGDocumentChunk
from api.rag_service import build_rag_service
from config.settings import get_settings

def test_session_scoped_rag():
    db_path = "data/test_rag_session_scope.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    init_db(db_path)
    db = get_session()

    try:
        # 1. Create documents for session_A
        doc_a = RAGDocument(
            id="doc_a",
            filename="document_session_a.txt",
            file_type="txt",
            content="Context specific to Session A and user account settings.",
            pipeline="chat",
            session_id="session_A"
        )
        db.add(doc_a)
        db.flush()

        chunk_a = RAGDocumentChunk(
            id="chunk_a",
            document_id=doc_a.id,
            section_name="General",
            chunk_index=0,
            text="Context specific to Session A and user account settings.",
            meta_json="{}"
        )
        db.add(chunk_a)

        # 2. Create documents for session_B
        doc_b = RAGDocument(
            id="doc_b",
            filename="document_session_b.txt",
            file_type="txt",
            content="Context specific to Session B and sales conversions details.",
            pipeline="chat",
            session_id="session_B"
        )
        db.add(doc_b)
        db.flush()

        chunk_b = RAGDocumentChunk(
            id="chunk_b",
            document_id=doc_b.id,
            section_name="General",
            chunk_index=0,
            text="Context specific to Session B and sales conversions details.",
            meta_json="{}"
        )
        db.add(chunk_b)

        # 3. Create document without session (legacy)
        doc_legacy = RAGDocument(
            id="doc_legacy",
            filename="document_legacy.txt",
            file_type="txt",
            content="Legacy documentation about older features.",
            pipeline="chat",
            session_id=None
        )
        db.add(doc_legacy)
        db.flush()

        chunk_legacy = RAGDocumentChunk(
            id="chunk_legacy",
            document_id=doc_legacy.id,
            section_name="General",
            chunk_index=0,
            text="Legacy documentation about older features.",
            meta_json="{}"
        )
        db.add(chunk_legacy)
        db.commit()

        # Build RAG Service
        rag_service = build_rag_service()

        # Setup mock vector store behavior (returns all chunk IDs)
        mock_vs = MagicMock()
        mock_vs.search.return_value = [("chunk_a", 0.95), ("chunk_b", 0.90), ("chunk_legacy", 0.85)]
        rag_service.chat.vector_store = mock_vs
        rag_service.code.vector_store = mock_vs

        # Setup mock embedder behavior
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.zeros(384)
        rag_service.chat.embedder = mock_emb
        rag_service.code.embedder = mock_emb

        # Test Case 1: Sessionless Retrieval Rule (session_id is None)
        # Should return empty and keep RAG OFF.
        results_none = rag_service.retrieve("context", db, session_id=None)
        assert len(results_none) == 0, f"Expected 0 results for sessionless search, got {len(results_none)}"

        # Test Case 2: Session isolation - session_A scope
        # Should only retrieve chunk_a.
        results_a = rag_service.retrieve("context", db, session_id="session_A")
        assert len(results_a) == 1, f"Expected exactly 1 result for session_A, got {len(results_a)}"
        assert results_a[0].chunk_id == "chunk_a", f"Expected chunk_a, got {results_a[0].chunk_id}"

        # Test Case 3: Session isolation - session_B scope
        # Should only retrieve chunk_b.
        results_b = rag_service.retrieve("context", db, session_id="session_B")
        assert len(results_b) == 1, f"Expected exactly 1 result for session_B, got {len(results_b)}"
        assert results_b[0].chunk_id == "chunk_b", f"Expected chunk_b, got {results_b[0].chunk_id}"

        # Test Case 4: Non-existent session
        # Should return empty since no documents match.
        results_empty = rag_service.retrieve("context", db, session_id="session_unknown")
        assert len(results_empty) == 0, f"Expected 0 results for unknown session, got {len(results_empty)}"

        # Test Case 5: Verify build_context with session isolation
        context_a = rag_service.build_context("context", db, session_id="session_A")
        assert "Session A" in context_a, "Context A should contain text from chunk_a"
        assert "Session B" not in context_a, "Context A should not contain text from chunk_b"

        context_none = rag_service.build_context("context", db, session_id=None)
        assert context_none == "", "Sessionless build_context should return empty string"

        print("ALL ACTIVE RETRIEVAL SCOPE TESTS PASSED SUCCESSFULLY!")

    finally:
        db.close()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass

if __name__ == "__main__":
    test_session_scoped_rag()
