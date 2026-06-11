"""
AS Core — Project Manager (Etapa 5.1)

Encapsulates all logic for managing projects, project-scoped chats,
document associations, and default project migrations.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session

from api.project_models import Project, ProjectChat, ProjectDocument, ProjectChatMessage
from api.memory_models import MemoryVariable, MemoryTask, MemoryObservation
from api.rag_models import RAGDocument, RAGDocumentChunk

logger = logging.getLogger("as-code.runtime.projects.manager")


class ProjectManager:
    """
    Desecoupled service layer to manage projects, chats and documents.
    """

    def ensure_default_project(self, db: Session) -> Project:
        """
        Guarantees that a default project with slug 'general' exists.
        Migrates any orphaned chat sessions (from memory tables) to this project.
        """
        try:
            general_proj = db.query(Project).filter(Project.slug == "general").first()
            if not general_proj:
                logger.info("[PROJECTS] Creating default General project...")
                general_proj = Project(
                    id=str(uuid.uuid4()),
                    slug="general",
                    name="General",
                    description="Proyecto por defecto de AS Core"
                )
                db.add(general_proj)
                db.commit()
                db.refresh(general_proj)

            # Find all unique session_ids from memory tables
            s1 = db.query(MemoryVariable.session_id).distinct().all()
            s2 = db.query(MemoryTask.session_id).distinct().all()
            s3 = db.query(MemoryObservation.session_id).distinct().all()

            all_sessions = set(
                r[0] for r in (s1 + s2 + s3) if r[0]
            )

            # Associate any orphaned sessions with General
            migrated_count = 0
            for session_id in all_sessions:
                exists = db.query(ProjectChat).filter(ProjectChat.session_id == session_id).first()
                if not exists:
                    title = f"Chat {session_id[:8]}" if len(session_id) > 8 else f"Chat {session_id}"
                    chat = ProjectChat(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        project_id=general_proj.id,
                        title=title
                    )
                    db.add(chat)
                    migrated_count += 1

            if migrated_count > 0:
                db.commit()
                logger.info(f"[PROJECTS] Associated {migrated_count} orphaned sessions with General project.")

            return general_proj
        except Exception as e:
            db.rollback()
            logger.error(f"[PROJECTS] Error ensuring default project: {e}")
            raise

    def create_project(self, db: Session, name: str, slug: str, description: Optional[str] = None) -> Project:
        """
        Creates a new project.
        """
        # Ensure unique slug
        existing = db.query(Project).filter(Project.slug == slug).first()
        if existing:
            raise ValueError(f"Project slug '{slug}' already exists.")

        project = Project(
            id=str(uuid.uuid4()),
            name=name,
            slug=slug,
            description=description
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        return project

    def get_project(self, db: Session, project_id: str) -> Optional[Project]:
        """
        Retrieves a project by ID.
        """
        return db.query(Project).filter(Project.id == project_id).first()

    def get_project_by_slug(self, db: Session, slug: str) -> Optional[Project]:
        """
        Retrieves a project by Slug.
        """
        return db.query(Project).filter(Project.slug == slug).first()

    def list_projects(self, db: Session) -> List[Project]:
        """
        Lists all projects.
        """
        return db.query(Project).order_by(Project.created_at.desc()).all()

    def rename_project(self, db: Session, project_id: str, new_name: str) -> Optional[Project]:
        """
        Renames a project. The slug is NOT changed (it serves as a stable identifier).
        """
        proj = self.get_project(db, project_id)
        if not proj:
            return None
        proj.name = new_name
        db.commit()
        db.refresh(proj)
        return proj

    def delete_project(self, db: Session, project_id: str) -> bool:
        """
        Deletes a project and ALL its chats, messages and session-scoped documents.
        The default 'general' project cannot be deleted.
        """
        proj = self.get_project(db, project_id)
        if not proj:
            return False

        if proj.slug == "general":
            raise ValueError("The default 'General' project cannot be deleted.")

        # Get all chats in the project
        chats = self.list_chats(db, project_id)
        for chat in chats:
            self.delete_chat_by_session(db, chat.session_id)

        # Remove residual ProjectDocument associations for this project
        db.query(ProjectDocument).filter(ProjectDocument.project_id == project_id).delete()

        db.delete(proj)
        db.commit()
        return True

    def create_chat(self, db: Session, project_id: str, session_id: str, title: Optional[str] = None) -> ProjectChat:
        """
        Creates a new chat mapped to a project.
        """
        # Ensure project exists
        proj = self.get_project(db, project_id)
        if not proj:
            raise ValueError(f"Project with ID '{project_id}' does not exist.")

        # Ensure session_id is unique across all chats
        existing = db.query(ProjectChat).filter(ProjectChat.session_id == session_id).first()
        if existing:
            return existing

        chat = ProjectChat(
            id=str(uuid.uuid4()),
            session_id=session_id,
            project_id=project_id,
            title=title or "Nuevo Chat"
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return chat

    def list_chats(self, db: Session, project_id: str) -> List[ProjectChat]:
        """
        Lists all chats under a project.
        """
        return db.query(ProjectChat).filter(
            ProjectChat.project_id == project_id
        ).order_by(ProjectChat.created_at.desc()).all()

    def get_chat_by_session(self, db: Session, session_id: str) -> Optional[ProjectChat]:
        """
        Retrieves the chat associated with a session ID.
        """
        return db.query(ProjectChat).filter(ProjectChat.session_id == session_id).first()

    def delete_chat_by_session(self, db: Session, session_id: str) -> bool:
        """
        Deletes a chat session, its message history, and any RAG documents
        that were uploaded within that session.
        Does NOT touch working memory, shared project documents, or FAISS vectors.
        """
        # 1. Delete session-scoped RAG document chunks and documents
        try:
            session_docs = db.query(RAGDocument).filter(
                RAGDocument.session_id == session_id
            ).all()
            for doc in session_docs:
                # Delete chunks
                db.query(RAGDocumentChunk).filter(RAGDocumentChunk.document_id == doc.id).delete()
                # Remove from ProjectDocument associations
                db.query(ProjectDocument).filter(ProjectDocument.document_id == doc.id).delete()
                db.delete(doc)
        except Exception as e:
            logger.warning(f"[DELETE-CHAT] Could not delete session documents for {session_id}: {e}")

        # 2. Delete history messages
        db.query(ProjectChatMessage).filter(ProjectChatMessage.session_id == session_id).delete()

        # 3. Delete the chat record
        chat = self.get_chat_by_session(db, session_id)
        if not chat:
            db.commit()
            return False

        db.delete(chat)
        db.commit()
        return True

    def associate_document(self, db: Session, project_id: str, document_id: str) -> ProjectDocument:
        """
        Maps a document to a project. Unique constraint prevents duplicates.
        """
        # Ensure project exists
        proj = self.get_project(db, project_id)
        if not proj:
            raise ValueError(f"Project with ID '{project_id}' does not exist.")

        # Check if already associated
        existing = db.query(ProjectDocument).filter_by(
            project_id=project_id, document_id=document_id
        ).first()
        if existing:
            return existing

        assoc = ProjectDocument(
            project_id=project_id,
            document_id=document_id
        )
        db.add(assoc)
        db.commit()
        db.refresh(assoc)
        return assoc

    def disassociate_document(self, db: Session, project_id: str, document_id: str) -> bool:
        """
        Removes association of a document with a project.
        Does not physically delete the document.
        """
        assoc = db.query(ProjectDocument).filter_by(
            project_id=project_id, document_id=document_id
        ).first()
        if not assoc:
            return False
        db.delete(assoc)
        db.commit()
        return True

    def rename_chat(self, db: Session, session_id: str, new_title: str) -> Optional[ProjectChat]:
        """
        Renames the chat associated with a session ID.
        """
        chat = self.get_chat_by_session(db, session_id)
        if not chat:
            return None
        chat.title = new_title
        db.commit()
        db.refresh(chat)
        return chat

    def delete_chat_by_session(self, db: Session, session_id: str) -> bool:
        """
        Deletes a chat session and all its messages.
        Does NOT touch working memory, documents, or vectors.
        """
        # Delete history messages first
        db.query(ProjectChatMessage).filter(ProjectChatMessage.session_id == session_id).delete()
        
        # Delete the chat record
        chat = self.get_chat_by_session(db, session_id)
        if not chat:
            db.commit()
            return False
        
        db.delete(chat)
        db.commit()
        return True

    def add_chat_message(self, db: Session, session_id: str, role: str, content: str) -> ProjectChatMessage:
        """
        Appends a chat message to the history of a session.
        """
        msg = ProjectChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    def list_chat_messages(self, db: Session, session_id: str) -> List[ProjectChatMessage]:
        """
        Retrieves all messages for a session ID in chronological order.
        """
        return db.query(ProjectChatMessage).filter(
            ProjectChatMessage.session_id == session_id
        ).order_by(ProjectChatMessage.created_at.asc()).all()

