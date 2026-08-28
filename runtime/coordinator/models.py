from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from enum import Enum

class ResetReason(str, Enum):
    """Razones de reset (determinísticas)."""
    EXPLICIT = "explicit_reset"      # Usuario ejecutó /reset
    LENGTH_OVERFLOW = "length_overflow"  # Query > max_chars
    WORD_THRESHOLD = "word_threshold"    # Query > max_words
    TOPIC_CHANGE = "topic_change"        # Cambio de entidad detectado
    NONE = "none"

class SessionSnapshot(BaseModel):
    """Estado minimal de sesión (sin ML, sin embeddings)."""
    session_id: str
    turn_number: int
    rag_query_stack: List[str] = Field(default_factory=list)  # Max 5 items
    language_history: List[Tuple[str, int]] = Field(default_factory=list)  # Max 5 items
    max_query_chars: int = 400      # Límite absoluto de chars
    max_query_words: int = 50       # Límite absoluto de palabras
    merge_max_ratio: float = 0.7    # % máximo de carryover

    @property
    def last_rag_query(self) -> Optional[str]:
        """Query anterior (O(1))."""
        return self.rag_query_stack[-1] if self.rag_query_stack else None

    @property
    def last_language(self) -> str:
        """Idioma del último turno (O(1))."""
        if self.language_history:
            return self.language_history[-1][0]
        return "ES"  # Default fallback

    def push_query(self, query: str) -> None:
        """Agregar query al stack (mantiene max 5)."""
        self.rag_query_stack.append(query)
        if len(self.rag_query_stack) > 5:
            self.rag_query_stack.pop(0)

    def push_language(self, lang: str, turn: int) -> None:
        """Agregar idioma al histórico (mantiene max 5)."""
        self.language_history.append((lang, turn))
        if len(self.language_history) > 5:
            self.language_history.pop(0)

class WorkflowState(BaseModel):
    objective: Optional[str] = None
    current_phase: Optional[str] = None
    current_focus: Optional[str] = None
    active_skill: Optional[str] = None

class CoordinatorDecision(BaseModel):
    resolved_skill: Optional[str] = None
    suggested_skills: List[str] = []
    workflow_state: WorkflowState
    runtime_context: str

class RuntimeContract(BaseModel):
    """Contrato de ejecución: minimal, determinístico."""
    request_id: str
    session_id: str
    model_id: str
    user_message: str
    previous_user_message: Optional[str] = None
    manual_skill: Optional[str] = None
    preset_name: Optional[str] = None
    timestamp: float
    
    # Deterministic Continuity extension fields
    snapshot: Optional[SessionSnapshot] = None
    language_confidence_threshold: int = 2
    explicit_reset: bool = False

class RAGHitMetadata(BaseModel):
    document_id: int
    filename: str
    section: Optional[str] = None
    score: float

class ContextManifest(BaseModel):
    contract_id: str
    active_skill: Optional[str] = None
    workflow_state: WorkflowState
    suggested_skills: List[str] = []
    
    # RAG metadata (sin contenido de texto gigante)
    rag_enabled: bool
    rag_query: Optional[str] = None
    rag_hits: List[RAGHitMetadata] = []
    
    # Conteos de Working Memory
    memory_variables_count: int = 0
    memory_tasks_count: int = 0
    memory_observations_count: int = 0
    
    # Presupuesto físico del prompt (caracteres)
    char_budget: int = 16000
    char_count: int = 0
    
    # Prompt compilado definitivo
    system_prompt_snapshot: str
    
    # Decisión de continuidad determinística
    continuity_decision: Optional[ContinuityDecision] = None
    
    # Capability Gate flag
    capability_gate_open: bool = False

class ContinuityDecision(BaseModel):
    """Resultado de decisión de continuidad (determinístico, serializable)."""
    final_rag_query: str
    detected_language: str
    reset_triggered: bool
    reset_reason: ResetReason
    merge_ratio: float  # 0.0 = no merge, 1.0 = 100% carryover
    
    # Debug info
    query_word_count: int
    query_char_count: int
    language_confidence: int  # Cantidad de palabras clave detectadas
