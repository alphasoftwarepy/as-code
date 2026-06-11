import sys
import os
import time

# Add root workspace to PYTHONPATH
sys.path.append("c:/as-code")

from api.database import init_db, get_session
from api.project_models import Project, ProjectChat, ProjectDocument
from api.memory_models import MemoryVariable
from api.rag_models import RAGDocument
from runtime.projects.manager import ProjectManager


def test_project_crud():
    # Setup test DB
    db_path = "data/test_projects.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    init_db(db_path)
    db = get_session()
    
    try:
        manager = ProjectManager()
        
        # 1. Test ensure_default_project creates 'general' project
        default_proj = manager.ensure_default_project(db)
        assert default_proj is not None
        assert default_proj.slug == "general"
        assert default_proj.name == "General"

        # 2. Test create new project
        proj = manager.create_project(db, name="E-Commerce", slug="e-commerce", description="Tienda online")
        assert proj.id is not None
        assert proj.slug == "e-commerce"
        assert proj.name == "E-Commerce"

        # 3. Test list projects
        projects = manager.list_projects(db)
        assert len(projects) == 2  # General + E-Commerce
        
        # 4. Test delete project
        success = manager.delete_project(db, proj.id)
        assert success is True
        assert manager.get_project(db, proj.id) is None
        
        # 5. Test default project cannot be deleted
        try:
            manager.delete_project(db, default_proj.id)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    finally:
        bind = db.bind
        db.close()
        if bind:
            bind.dispose()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_project_chats():
    # Setup test DB
    db_path = "data/test_project_chats.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    init_db(db_path)
    db = get_session()
    
    try:
        manager = ProjectManager()
        default_proj = manager.ensure_default_project(db)

        # 1. Create a chat session
        session_id = "sess_abc_123"
        chat = manager.create_chat(db, project_id=default_proj.id, session_id=session_id, title="Test Chat")
        assert chat.id is not None
        assert chat.session_id == session_id
        assert chat.project_id == default_proj.id
        assert chat.title == "Test Chat"

        # 2. Retrieve chat by session
        retrieved = manager.get_chat_by_session(db, session_id)
        assert retrieved is not None
        assert retrieved.id == chat.id

        # 3. List chats
        chats = manager.list_chats(db, default_proj.id)
        assert len(chats) == 1
        assert chats[0].id == chat.id

        # 4. Delete chat
        success = manager.delete_chat(db, chat.id)
        assert success is True
        assert manager.get_chat_by_session(db, session_id) is None

    finally:
        bind = db.bind
        db.close()
        if bind:
            bind.dispose()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_orphaned_session_migration():
    # Setup test DB
    db_path = "data/test_project_migration.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    init_db(db_path)
    db = get_session()
    
    try:
        # Create an orphaned session in memory tables before ProjectManager is initialized
        orphaned_session = "orphaned_session_xyz"
        var = MemoryVariable(
            session_id=orphaned_session,
            key="test_key",
            value="test_val"
        )
        db.add(var)
        db.commit()

        manager = ProjectManager()
        # Initialize default project, which triggers migration of orphaned sessions
        default_proj = manager.ensure_default_project(db)

        # Verify the orphaned session was migrated
        chat = manager.get_chat_by_session(db, orphaned_session)
        assert chat is not None
        assert chat.project_id == default_proj.id
        assert orphaned_session[:8] in chat.title

    finally:
        bind = db.bind
        db.close()
        if bind:
            bind.dispose()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_project_document_association():
    # Setup test DB
    db_path = "data/test_project_docs.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    init_db(db_path)
    db = get_session()
    
    try:
        manager = ProjectManager()
        default_proj = manager.ensure_default_project(db)

        # Create dummy document record in RAGDocument
        doc_id = "doc_test_123"
        doc = RAGDocument(
            id=doc_id,
            filename="proposal.pdf",
            file_type="pdf",
            content="dummy text",
            source="local",
            pipeline="chat",
            session_id="session_xyz"
        )
        db.add(doc)
        db.commit()

        # Associate doc with project
        assoc = manager.associate_document(db, project_id=default_proj.id, document_id=doc_id)
        assert assoc.project_id == default_proj.id
        assert assoc.document_id == doc_id

        # Try associating again (should ignore/return existing due to unique constraints)
        assoc2 = manager.associate_document(db, project_id=default_proj.id, document_id=doc_id)
        assert assoc2.project_id == default_proj.id

        # Disassociate doc from project
        success = manager.disassociate_document(db, project_id=default_proj.id, document_id=doc_id)
        assert success is True

    finally:
        bind = db.bind
        db.close()
        if bind:
            bind.dispose()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_chat_history_and_actions():
    # Setup test DB
    db_path = "data/test_chat_history_actions.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass
            
    init_db(db_path)
    db = get_session()
    
    try:
        manager = ProjectManager()
        default_proj = manager.ensure_default_project(db)
        session_id = "sess_history_1"

        # 1. Create a chat session
        chat = manager.create_chat(db, project_id=default_proj.id, session_id=session_id, title="Nuevo Chat")
        assert chat is not None

        # 2. Add message history
        msg1 = manager.add_chat_message(db, session_id, "user", "Hola, ¿cómo estás?")
        time.sleep(0.1) # ensure created_at ordering differs slightly
        msg2 = manager.add_chat_message(db, session_id, "assistant", "¡Hola! Estoy muy bien, ¿y tú?")
        
        assert msg1.id is not None
        assert msg1.session_id == session_id
        assert msg1.role == "user"
        assert msg1.content == "Hola, ¿cómo estás?"

        # 3. Retrieve and verify message order
        messages = manager.list_chat_messages(db, session_id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[0].created_at < messages[1].created_at

        # 4. Rename chat
        renamed = manager.rename_chat(db, session_id, "Conversacion de prueba")
        assert renamed is not None
        assert renamed.title == "Conversacion de prueba"
        assert manager.get_chat_by_session(db, session_id).title == "Conversacion de prueba"

        # 5. Delete chat by session (should also delete messages)
        del_success = manager.delete_chat_by_session(db, session_id)
        assert del_success is True
        assert manager.get_chat_by_session(db, session_id) is None
        assert len(manager.list_chat_messages(db, session_id)) == 0

    finally:
        bind = db.bind
        db.close()
        if bind:
            bind.dispose()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


if __name__ == "__main__":
    test_project_crud()
    test_project_chats()
    test_orphaned_session_migration()
    test_project_document_association()
    test_chat_history_and_actions()
    print("ALL PROJECT SYSTEM TESTS PASSED SUCCESSFULLY!")

