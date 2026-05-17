"""
AS Code — Document API Routes

Endpoints:
  POST   /api/documents/session        → crear sesión
  POST   /api/documents/upload         → subir archivo
  GET    /api/documents/{session_id}   → listar documentos
  DELETE /api/documents/{session_id}   → limpiar sesión
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
import logging

from api.document_service import DocumentService, get_document_service

logger = logging.getLogger(__name__)

documents_router = APIRouter(prefix="/api/documents", tags=["documents"])


# ── Modelos de respuesta ────────────────────────────────────────

class CreateSessionResponse(BaseModel):
    session_id: str


class DocumentInfo(BaseModel):
    filename: str
    file_type: str
    char_count: int


class UploadResponse(BaseModel):
    filename: str
    file_type: str
    char_count: int
    message: str


class DocumentsListResponse(BaseModel):
    session_id: str
    document_count: int
    documents: list[DocumentInfo]
    context_preview: str  # Primeros 300 chars


# ── Endpoints ───────────────────────────────────────────────────

@documents_router.post("/session", response_model=CreateSessionResponse)
async def create_document_session(
    service: DocumentService = Depends(get_document_service)
):
    """Crea nueva sesión de documentos para esta conversación."""
    session_id = service.create_session()
    return CreateSessionResponse(session_id=session_id)


@documents_router.post("/upload", response_model=UploadResponse)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    """
    Sube y parsea un documento.

    Soportados: .txt, .pdf, .docx

    Query params:
      - session_id: ID de sesión (obtén con POST /api/documents/session)
    """
    try:
        doc = service.upload_and_parse(file, session_id)
        return UploadResponse(
            filename=doc.filename,
            file_type=doc.file_type,
            char_count=doc.char_count,
            message=f"✓ {doc.filename} cargado ({doc.file_type})",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error en upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error parseando archivo: {e}")


@documents_router.get("/{session_id}", response_model=DocumentsListResponse)
async def list_documents(
    session_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """Lista documentos de una sesión."""
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Sesión no existe: {session_id}")

    docs = [
        DocumentInfo(
            filename=d.filename,
            file_type=d.file_type,
            char_count=d.char_count,
        )
        for d in session.documents
    ]

    context = session.get_context(max_chars=300)

    return DocumentsListResponse(
        session_id=session_id,
        document_count=len(docs),
        documents=docs,
        context_preview=context[:300] + "..." if len(context) > 300 else context,
    )


@documents_router.delete("/{session_id}")
async def clear_session(
    session_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """Limpia sesión y elimina archivos."""
    service.clear_session(session_id)
    return {"message": f"Sesión {session_id} limpiada"}
