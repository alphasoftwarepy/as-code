import sys
import os
import time
from unittest.mock import MagicMock
import numpy as np

sys.path.append("c:/as-code")

# Mock modules before importing rag_service
import api.embedder_service
import api.vector_store_service
api.embedder_service.get_embedder = MagicMock()
api.vector_store_service.get_vector_store = MagicMock()

from api.database import init_db, get_session
from api.rag_models import RAGDocument, RAGDocumentChunk
from api.rag_service import build_rag_service

def test_ambiguous_query_prioritization():
    db_path = "data/test_rag_ambiguous.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    init_db(db_path)
    db = get_session()

    try:
        # Create an old document
        old_doc = RAGDocument(
            id="doc_old",
            filename="historical_marketing.txt",
            file_type="txt",
            content="This is historical marketing content about campaigns and target prospects.",
            pipeline="chat"
        )
        db.add(old_doc)
        db.flush()

        old_chunk = RAGDocumentChunk(
            id="chunk_old",
            document_id=old_doc.id,
            section_name="Introduction",
            chunk_index=0,
            text="This is historical marketing content about campaigns and target prospects.",
            meta_json="{}"
        )
        db.add(old_chunk)

        # Sleep slightly to guarantee different timestamp
        time.sleep(0.1)

        # Create a new document (most recently loaded)
        new_doc = RAGDocument(
            id="doc_new",
            filename="new_python_fix.py",
            file_type="py",
            content="def calculate_vram_limit(): return 8 * 1024 # Newly uploaded Python file.",
            pipeline="code"
        )
        db.add(new_doc)
        db.flush()

        new_chunk = RAGDocumentChunk(
            id="chunk_new",
            document_id=new_doc.id,
            section_name="Main",
            chunk_index=0,
            text="def calculate_vram_limit(): return 8 * 1024 # Newly uploaded Python file.",
            meta_json="{}"
        )
        db.add(new_chunk)
        db.commit()

        # Build RAG Service (dependencies mocked)
        rag_service = build_rag_service()

        # Setup mock vector store behavior
        mock_vs = MagicMock()
        mock_vs.search.return_value = [("chunk_old", 0.9), ("chunk_new", 0.8)]
        rag_service.chat.vector_store = mock_vs
        rag_service.code.vector_store = mock_vs

        # Setup mock embedder behavior
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = np.zeros(384)
        rag_service.chat.embedder = mock_emb
        rag_service.code.embedder = mock_emb

        # Execute normal query (should search both documents)
        results_normal = rag_service.retrieve("campaigns", db, top_k=2)
        assert len(results_normal) > 0, "Normal search should find matches"
        assert any(c.chunk_id == "chunk_old" for c in results_normal)

        # Execute ambiguous query (should only return chunks belonging to doc_new)
        results_ambiguous = rag_service.retrieve("de que trata esto", db, top_k=2)
        assert len(results_ambiguous) > 0, "Ambiguous search should find matches"
        
        # Verify that all returned chunks are from doc_new
        for chunk in results_ambiguous:
            assert chunk.document_id == "doc_new", f"Expected doc_new, got {chunk.document_filename}"

        print("RAG AMBIGUOUS QUERY TEST PASSED SUCCESSFULLY!")

    finally:
        db.close()
        # Clean db file
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass

if __name__ == "__main__":
    test_ambiguous_query_prioritization()
