"""
AS Code — RAG Database

SQLite persistence for document and chunk *metadata* only.
Embeddings are stored separately in FAISS for performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal: Optional[sessionmaker] = None


def init_db(db_path: str = "data/rag.db") -> None:
    """Initialize SQLite engine and create all tables."""
    global _engine, _SessionLocal

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )

    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Import here to avoid circular imports at module load time
    from api.rag_models import Base  # noqa: F401
    from api.memory_models import MemoryBase  # noqa: F401
    Base.metadata.create_all(bind=_engine)
    MemoryBase.metadata.create_all(bind=_engine)

    logger.info(f"RAG database initialized: {db_path}")


def get_db() -> Generator[Optional[Session], None, None]:
    """
    FastAPI dependency: yields a scoped DB session, or None if RAG is disabled.

    Callers must check `if db is None` before use.  This keeps the dependency
    safe to inject into all routes regardless of ENABLE_RAG_MODE.
    """
    if _SessionLocal is None:
        yield None
        return
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Session:
    """Direct session for background tasks (non-dependency usage)."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized — call init_db() at startup.")
    return _SessionLocal()
