"""
AS Code — Project API Routes (Etapa 5.1)

Endpoints scoped under /v1/projects for project and chat lifecycle management.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.database import get_db

logger = logging.getLogger("as-code.api.projects")

project_router = APIRouter(prefix="/v1/projects", tags=["Projects"])


# ── Request schemas ────────────────────────────────────────────

class ProjectCreatePayload(BaseModel):
    name: str = Field(..., min_length=1)
    slug: str = Field(..., min_length=1)
    description: Optional[str] = None


class ChatCreatePayload(BaseModel):
    session_id: str = Field(..., min_length=1)
    title: Optional[str] = None


class ChatRenamePayload(BaseModel):
    title: str = Field(..., min_length=1)



# ── Helper ─────────────────────────────────────────────────────

def get_manager(request: Request):
    """Retrieves the ProjectManager singleton from app state."""
    manager = getattr(request.app.state, "project_manager", None)
    if not manager:
        # Fallback if not initialized in lifespan
        from runtime.projects.manager import ProjectManager
        manager = ProjectManager()
        request.app.state.project_manager = manager
    return manager


# ── Endpoints ──────────────────────────────────────────────────

@project_router.get("", response_model=List[dict])
def list_projects(
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Lists all projects."""
    projects = manager.list_projects(db)
    return [p.to_dict() for p in projects]


@project_router.post("", response_model=dict)
def create_project(
    payload: ProjectCreatePayload,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Creates a new project."""
    try:
        project = manager.create_project(
            db,
            name=payload.name,
            slug=payload.slug,
            description=payload.description
        )
        return project.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@project_router.get("/{project_id}", response_model=dict)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Retrieves a single project by ID."""
    project = manager.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.to_dict()


@project_router.patch("/{project_id}", response_model=dict)
def rename_project(
    project_id: str,
    payload: ChatRenamePayload,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Renames a project."""
    proj = manager.rename_project(db, project_id, payload.title)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return proj.to_dict()


@project_router.delete("/{project_id}", response_model=dict)
def delete_project(
    project_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Deletes a project and all its chats, messages and session-scoped documents."""
    try:
        success = manager.delete_project(db, project_id)
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"status": "success", "message": f"Project '{project_id}' deleted."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



@project_router.get("/{project_id}/chats", response_model=List[dict])
def list_chats(
    project_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Lists all chats belonging to a project."""
    # Verify project exists
    project = manager.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    chats = manager.list_chats(db, project_id)
    return [c.to_dict() for c in chats]


@project_router.post("/{project_id}/chats", response_model=dict)
def create_chat(
    project_id: str,
    payload: ChatCreatePayload,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Creates a new chat session mapped to a project."""
    try:
        chat = manager.create_chat(
            db,
            project_id=project_id,
            session_id=payload.session_id,
            title=payload.title
        )
        return chat.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@project_router.post("/{project_id}/documents/{document_id}", response_model=dict)
def associate_document(
    project_id: str,
    document_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Maps a RAG document to a project."""
    try:
        assoc = manager.associate_document(db, project_id, document_id)
        return assoc.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error associating document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@project_router.delete("/{project_id}/documents/{document_id}", response_model=dict)
def disassociate_document(
    project_id: str,
    document_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Removes the association between a document and a project."""
    try:
        success = manager.disassociate_document(db, project_id, document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Association not found")
        return {"status": "success", "message": f"Document '{document_id}' unlinked from project '{project_id}'."}
    except Exception as e:
        logger.error(f"Error disassociating document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@project_router.get("/{project_id}/chats/{session_id}", response_model=dict)
def get_chat(
    project_id: str,
    session_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Retrieves chat details by session ID."""
    chat = manager.get_chat_by_session(db, session_id)
    if not chat or chat.project_id != project_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat.to_dict()


@project_router.patch("/{project_id}/chats/{session_id}", response_model=dict)
def rename_chat(
    project_id: str,
    session_id: str,
    payload: ChatRenamePayload,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Renames an existing chat session."""
    chat = manager.rename_chat(db, session_id, payload.title)
    if not chat or chat.project_id != project_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat.to_dict()


@project_router.delete("/{project_id}/chats/{session_id}", response_model=dict)
def delete_chat(
    project_id: str,
    session_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Deletes a chat session and its message history."""
    success = manager.delete_chat_by_session(db, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "success", "message": f"Chat '{session_id}' deleted."}


@project_router.get("/{project_id}/chats/{session_id}/messages", response_model=List[dict])
def get_chat_messages(
    project_id: str,
    session_id: str,
    db: Session = Depends(get_db),
    manager = Depends(get_manager)
):
    """Lists all historic messages for a chat session in chronological order."""
    chat = manager.get_chat_by_session(db, session_id)
    if not chat or chat.project_id != project_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    messages = manager.list_chat_messages(db, session_id)
    return [m.to_dict() for m in messages]

