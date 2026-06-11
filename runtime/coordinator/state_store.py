import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional
from sqlalchemy.orm import Session
from api.memory_models import MemoryVariable
from runtime.coordinator.models import SessionSnapshot, ContinuityDecision, ResetReason

logger = logging.getLogger("as-code.runtime.coordinator.state_store")

@dataclass
class SessionStateRecord:
    """Registro único de estado (determinístico, serializable)."""
    session_id: str
    turn_number: int
    rag_query: str
    language: str
    reset_triggered: bool
    reset_reason: str
    merge_ratio: float
    request_id: str
    timestamp_ms: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'SessionStateRecord':
        data = json.loads(json_str)
        return cls(**data)


class LightweightStateStore:
    """
    Almacén de estado ultraligero que persiste stacks de consultas e idiomas
    en la tabla existente MemoryVariable para evitar transacciones complejas.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session

    def load_session_state(self, session_id: str) -> SessionSnapshot:
        """
        Carga el snapshot de sesión desde MemoryVariable.
        Si no existen registros, devuelve un snapshot limpio.
        """
        try:
            rag_var = self.db.query(MemoryVariable).filter_by(
                session_id=session_id, key="wf_last_rag_queries"
            ).first()
            
            lang_var = self.db.query(MemoryVariable).filter_by(
                session_id=session_id, key="wf_last_languages"
            ).first()

            rag_query_stack = json.loads(rag_var.value) if rag_var else []
            language_history = json.loads(lang_var.value) if lang_var else []
            
            # Asegurarse de que el formato coincida con Tuple[str, int] para language_history
            formatted_lang_history = []
            for item in language_history:
                if isinstance(item, list) and len(item) == 2:
                    formatted_lang_history.append((item[0], item[1]))
                elif isinstance(item, tuple) and len(item) == 2:
                    formatted_lang_history.append(item)

            return SessionSnapshot(
                session_id=session_id,
                turn_number=len(rag_query_stack),
                rag_query_stack=rag_query_stack,
                language_history=formatted_lang_history,
                max_query_chars=400,
                max_query_words=50,
                merge_max_ratio=0.7
            )
        except Exception as e:
            logger.error(f"[STATE-STORE] Fallo al cargar estado de sesión (degradando a limpio): {e}")
            return SessionSnapshot(
                session_id=session_id,
                turn_number=0,
                rag_query_stack=[],
                language_history=[],
                max_query_chars=400,
                max_query_words=50,
                merge_max_ratio=0.7
            )

    def persist_decision(
        self,
        session_id: str,
        turn_number: int,
        final_rag_query: str,
        language: str,
        decision: ContinuityDecision,
        request_id: str
    ) -> bool:
        """
        Guarda la decisión y el nuevo estado (stacks) en la base de datos de manera atómica.
        """
        try:
            from runtime.memory.manager import WorkingMemoryManager
            mem_mgr = WorkingMemoryManager()
            
            # Cargar snapshot actual para añadir nuevos elementos
            snapshot = self.load_session_state(session_id)
            snapshot.push_query(final_rag_query)
            snapshot.push_language(language, turn_number)
            
            # Persistir stacks serializados en JSON
            mem_mgr.set_variable(self.db, session_id, "wf_last_rag_queries", json.dumps(snapshot.rag_query_stack))
            mem_mgr.set_variable(self.db, session_id, "wf_last_languages", json.dumps(snapshot.language_history))
            
            # Persistir valores directos sencillos para lectura rápida legacy de skills / prompt assembly
            mem_mgr.set_variable(self.db, session_id, "wf_last_rag_query", final_rag_query)
            mem_mgr.set_variable(self.db, session_id, "wf_last_language", language)
            
            # Crear y guardar registro log detallado para auditoría
            timestamp_ms = int(time.time() * 1000)
            record = SessionStateRecord(
                session_id=session_id,
                turn_number=turn_number,
                rag_query=final_rag_query,
                language=language,
                reset_triggered=decision.reset_triggered,
                reset_reason=decision.reset_reason.value,
                merge_ratio=decision.merge_ratio,
                request_id=request_id,
                timestamp_ms=timestamp_ms
            )
            mem_mgr.set_variable(
                self.db, 
                session_id, 
                f"wf_decision_log_t{turn_number}", 
                record.to_json()
            )
            
            return True
        except Exception as e:
            logger.error(f"[STATE-STORE] Error al persistir decisión de continuidad: {e}")
            return False


class PendingSkillStore:
    """Encapsulates session pending skill states in process memory."""
    def __init__(self):
        self._store = {}

    def register_pending(self, session_id: str, capability_id: str) -> None:
        self._store[session_id] = capability_id

    def get_pending(self, session_id: str) -> Optional[str]:
        return self._store.get(session_id)

    def clear_pending(self, session_id: str) -> None:
        self._store.pop(session_id, None)


_global_pending_skill_store = PendingSkillStore()


def get_pending_skill_store() -> PendingSkillStore:
    """Access the global PendingSkillStore singleton."""
    return _global_pending_skill_store

