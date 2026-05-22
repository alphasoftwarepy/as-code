"""
AS Code — RAG API Routes

Endpoints:
  POST   /api/rag/documents/upload      Upload & ingest document
  GET    /api/rag/documents             List all documents
  DELETE /api/rag/documents/{doc_id}    Delete document + chunks + vectors
  POST   /api/rag/retrieve              Debug: raw chunk retrieval
  POST   /api/rag/context               Debug: build NotebookLM context preview
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from api.database import get_db
from api.document_parser import parse_document
from api.rag_models import RAGDocument
from api.vector_store_service import get_vector_store

logger = logging.getLogger(__name__)

rag_router = APIRouter(prefix="/api/rag", tags=["RAG"])


def _require_db(db: Optional[Session]) -> Session:
    """Guard: raise 503 if DB session is None (RAG not initialized)."""
    if db is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG database not initialized. "
                "Ensure ASCODE_ENABLE_RAG_MODE=true in .env and restart the server."
            ),
        )
    return db

from config.settings import get_settings

_UPLOADS_DIR = Path(get_settings().uploads_dir) / "rag"


def _get_rag_service(request: Request):
    """Retrieve RAGService from app.state."""
    rag = getattr(request.app.state, "rag_service", None)
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Set ASCODE_ENABLE_RAG_MODE=true and restart.",
        )
    return rag


# ── POST /api/rag/documents/upload ─────────────────────────────


@rag_router.post("/documents/upload", summary="Upload & ingest document for RAG")
async def upload_rag_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    pipeline: str = "chat",        # "chat" | "code"
    db: Optional[Session] = Depends(get_db),
):
    """
    Upload a document and start async RAG ingest.

    Pipeline selection:
      - **chat**  — conversational docs (txt, md, pdf, docx)
      - **code**  — source code files (py, js, ts, go, …)

    Returns immediately; chunking + embedding runs in background.
    """
    # Guard: fail fast with a clear 503 if RAG DB isn't initialized
    db = _require_db(db)
    rag_service = _get_rag_service(request)

    # Validate extension
    suffix = Path(file.filename).suffix.lower().lstrip(".")
    if not suffix:
        raise HTTPException(status_code=400, detail="Cannot determine file type.")

    # Save to disk
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = _UPLOADS_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    logger.info(
        f"[RAG-UPLOAD] file={file.filename!r} | "
        f"size_bytes={len(content)} | pipeline={pipeline}"
    )

    # Parse text
    try:
        parsed = parse_document(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    logger.info(
        f"[RAG-PARSE] filename={parsed.filename!r} | "
        f"file_type={parsed.file_type} | chars={len(parsed.text)}"
    )

    # Persist document record to SQLite
    doc = RAGDocument(
        filename=parsed.filename,
        file_type=parsed.file_type,
        content=parsed.text,
        source="local",
        pipeline=pipeline,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    logger.info(
        f"[RAG-SAVED] doc_id={doc.id!r} | filename={doc.filename!r} | "
        f"pipeline={pipeline} | starting_background_ingest=True"
    )

    # Async ingest (chunk → embed → FAISS) in background
    from api.database import get_session

    def _ingest(doc_id: str, pip: str) -> None:
        bg_db = get_session()
        try:
            logger.info(f"[RAG-BG-START] doc_id={doc_id!r} | pipeline={pip}")
            bg_doc = bg_db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
            if bg_doc:
                n = rag_service.process_document(bg_doc, bg_db)
                logger.info(
                    f"[RAG-BG-DONE] doc_id={doc_id!r} | "
                    f"filename={bg_doc.filename!r} | chunks={n}"
                )
            else:
                logger.error(f"[RAG-BG-ERROR] doc_id={doc_id!r} not found in DB")
        except Exception as exc:
            logger.error(
                f"[RAG-BG-ERROR] doc_id={doc_id!r} | error={exc!r}",
                exc_info=True,
            )
        finally:
            bg_db.close()

    background_tasks.add_task(_ingest, doc.id, pipeline)

    return {
        "status": "uploaded",
        "document_id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "pipeline": pipeline,
        "processing": True,
        "message": "Document saved. Chunking and embedding running in background.",
    }


# ── GET /api/rag/documents ─────────────────────────────────────


@rag_router.get("/documents", summary="List all RAG documents")
def list_rag_documents(db: Optional[Session] = Depends(get_db)):
    """Return all uploaded documents with chunk counts."""
    db = _require_db(db)
    docs = db.query(RAGDocument).order_by(RAGDocument.created_at.desc()).all()
    return {
        "count": len(docs),
        "documents": [d.to_dict() for d in docs],
    }


# ── DELETE /api/rag/documents ─────────────────────────────────


@rag_router.delete("/documents", summary="Delete all documents + chunks + vectors")
def delete_all_rag_documents(db: Optional[Session] = Depends(get_db)):
    """
    Delete all documents, their chunks from SQLite, and unmap vectors from FAISS.
    """
    db = _require_db(db)
    docs = db.query(RAGDocument).all()
    if not docs:
        return {
            "status": "deleted",
            "documents_removed": 0,
            "chunks_removed": 0,
            "message": "No documents found to delete."
        }

    all_chunk_ids = []
    for doc in docs:
        all_chunk_ids.extend([c.id for c in doc.chunks])

    # Unmap from FAISS (graceful)
    try:
        vs = get_vector_store()
        vs.remove_chunks(all_chunk_ids)
    except Exception as e:
        logger.warning(f"Could not unmap FAISS vectors: {e}")

    # Delete physical files from disk
    try:
        for doc in docs:
            file_path = _UPLOADS_DIR / doc.filename
            if file_path.exists():
                file_path.unlink()
    except Exception as e:
        logger.warning(f"Could not delete physical files: {e}")

    # Delete all documents from SQLite (cascade deletes chunks)
    for doc in docs:
        db.delete(doc)
    db.commit()

    logger.info(f"All documents deleted: {len(docs)} documents, {len(all_chunk_ids)} chunks unmapped")

    return {
        "status": "deleted",
        "documents_removed": len(docs),
        "chunks_removed": len(all_chunk_ids),
    }


# ── DELETE /api/rag/documents/{doc_id} ────────────────────────


@rag_router.delete("/documents/{doc_id}", summary="Delete document + chunks + vectors")
def delete_rag_document(
    doc_id: str,
    db: Optional[Session] = Depends(get_db),
):
    """
    Delete a document, its chunks from SQLite, and unmap vectors from FAISS.

    Note: FAISS IndexFlatIP does not support true deletion.
    Vectors are *unmapped* (will never be returned) but remain as stale slots.
    """
    db = _require_db(db)
    doc = db.query(RAGDocument).filter(RAGDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

    # Collect chunk IDs before deletion
    chunk_ids = [c.id for c in doc.chunks]

    # Unmap from FAISS (graceful — no crash if index not yet built)
    try:
        vs = get_vector_store()
        vs.remove_chunks(chunk_ids)
    except Exception as e:
        logger.warning(f"Could not unmap FAISS vectors: {e}")

    # Delete physical file from disk
    try:
        file_path = _UPLOADS_DIR / doc.filename
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning(f"Could not delete physical file: {e}")

    # Delete from SQLite (cascade deletes chunks)
    db.delete(doc)
    db.commit()

    logger.info(f"Document deleted: {doc_id} ({len(chunk_ids)} chunks unmapped)")

    return {
        "status": "deleted",
        "document_id": doc_id,
        "chunks_removed": len(chunk_ids),
    }


# ── POST /api/rag/retrieve ─────────────────────────────────────


@rag_router.post("/retrieve", summary="Debug: raw chunk retrieval")
def retrieve_chunks(
    request: Request,
    query: str,
    top_k: int = 5,
    pipeline: str = "chat",
    db: Optional[Session] = Depends(get_db),
):
    """
    Test endpoint: retrieve top-k chunks without LLM.
    Useful for tuning retrieval quality before chat integration.
    """
    db = _require_db(db)
    rag_service = _get_rag_service(request)

    chunks = rag_service.retrieve(query, db, top_k=top_k, pipeline=pipeline)

    return {
        "query": query,
        "pipeline": pipeline,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "document": c.document_filename,
                "section": c.section_name,
                "type": c.chunk_type,
                "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                "score": round(c.score, 4),
                "relevance_pct": c.relevance_pct,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
    }


# ── POST /api/rag/context ──────────────────────────────────────


@rag_router.post("/context", summary="Debug: build NotebookLM context preview")
def build_context_preview(
    request: Request,
    query: str,
    mode: str = "normal",     # normal | thinking | code
    pipeline: str = "chat",
    top_k: int = 5,
    db: Optional[Session] = Depends(get_db),
):
    """
    Preview the context that would be injected into the LLM prompt.
    Useful for prompt engineering and debugging retrieval quality.
    """
    db = _require_db(db)
    rag_service = _get_rag_service(request)

    context = rag_service.build_context(
        query=query,
        db=db,
        mode=mode,
        pipeline=pipeline,
        top_k=top_k,
    )

    return {
        "query": query,
        "mode": mode,
        "pipeline": pipeline,
        "context_length": len(context),
        "context": context,
    }
