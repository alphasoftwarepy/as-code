"""
AS Code — Document Service
Maneja upload, almacenamiento temporal, y sesión de documentos.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from api.document_parser import ParsedDocument, parse_document, truncate_context

logger = logging.getLogger(__name__)


@dataclass
class DocumentSession:
    """Sesión de documentos para una conversación."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    documents: List[ParsedDocument] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_document(self, doc: ParsedDocument):
        """Agregar documento a la sesión."""
        self.documents.append(doc)
        logger.info(f"Doc agregado a sesión {self.session_id}: {doc.filename}")

    def get_context(self, max_chars: int = 8000) -> str:
        """
        Construye contexto para inyectar en chat.

        Formato:
        ---DOCUMENTO: filename---
        [texto]

        ---DOCUMENTO: filename2---
        [texto]
        """
        if not self.documents:
            return ""

        context_parts = []
        for doc in self.documents:
            text = truncate_context(doc.text, max_chars=max_chars // len(self.documents))
            context_parts.append(
                f"---DOCUMENTO: {doc.filename} ({doc.file_type})---\n{text}"
            )

        return "\n\n".join(context_parts)

    def clear(self):
        """Limpiar sesión."""
        self.documents.clear()


class DocumentService:
    """
    Servicio de documentos para AS Code.

    Gestiona:
    - Upload de archivos
    - Parsing
    - Sesiones (docs por conversación)
    - Cleanup automático
    """

    def __init__(self, upload_dir: str = "data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        # Sesiones activas: {session_id -> DocumentSession}
        self.sessions: dict[str, DocumentSession] = {}

        logger.info(f"DocumentService iniciado. Upload dir: {self.upload_dir}")

    def create_session(self) -> str:
        """Crea nueva sesión y retorna session_id."""
        session = DocumentSession()
        self.sessions[session.session_id] = session
        logger.info(f"Sesión creada: {session.session_id}")
        return session.session_id

    def get_session(self, session_id: str) -> Optional[DocumentSession]:
        """Obtiene sesión."""
        return self.sessions.get(session_id)

    def save_uploaded_file(self, file_obj, session_id: str) -> str:
        """
        Guarda archivo subido temporalmente.

        Args:
            file_obj: objeto de archivo (FastAPI UploadFile)
            session_id: sesión destino

        Returns:
            ruta local del archivo guardado
        """
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(exist_ok=True)

        file_path = session_dir / file_obj.filename

        with open(file_path, "wb") as f:
            f.write(file_obj.file.read())

        logger.info(f"Archivo guardado: {file_path}")
        return str(file_path)

    def upload_and_parse(self, file_obj, session_id: str) -> ParsedDocument:
        """
        Upload + parse en una pasada.

        Args:
            file_obj: UploadFile de FastAPI
            session_id: sesión destino

        Returns:
            ParsedDocument listo
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Sesión no existe: {session_id}")

        # Guarda archivo
        file_path = self.save_uploaded_file(file_obj, session_id)

        # Parsea
        doc = parse_document(file_path)

        # Agrega a sesión
        session.add_document(doc)

        return doc

    def clear_session(self, session_id: str):
        """Limpia sesión y archivos."""
        session = self.sessions.pop(session_id, None)
        if not session:
            return

        session_dir = self.upload_dir / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
            logger.info(f"Sesión limpiada: {session_id}")

    def get_context(self, session_id: str, max_chars: int = 8000) -> str:
        """Obtiene contexto de documentos para inyectar en chat."""
        session = self.get_session(session_id)
        if not session:
            return ""
        return session.get_context(max_chars=max_chars)


# Instancia global (singleton)
_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Dependency injection para FastAPI."""
    global _service
    if _service is None:
        _service = DocumentService()
    return _service
